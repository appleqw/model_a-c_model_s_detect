import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from detect_model_s import detect_model_s_responses, plot_model_s_responses

matplotlib.use('TkAgg') # 或者 'Qt5Agg', 'WXAgg'

# 全局设置：解决中文显示问题
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常

#配置参数
sample_format = np.int16  # 样本格式
bytes_per_sample = sample_format().nbytes  # 每个样本的字节数(2字节)

def read_iqh_header(iqh_path):
    """读取IQ头文件，兼容无冒号的行和空行"""
    header = {}
    with open(iqh_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(':', 1)
            if len(parts) != 2:
                print(f"警告：头文件第{line_num}行无有效键值对，已跳过：{line}")
                continue
            key, value = parts
            key = key.strip()
            value = value.strip()
            if key in ['CenterFrequency', 'SampleRate', 'AcqBandwidth',
                       'DataScale', 'Scale', 'ReferenceLevel']:
                header[key] = float(value)
            elif key == 'NumberSamples':
                header[key] = int(value)
            else:
                header[key] = value
    # 检查关键参数
    required_keys = ['SampleRate', 'NumberSamples', 'DataScale']
    for key in required_keys:
        if key not in header:
            raise ValueError(f"头文件缺少关键参数：{key}，请检查文件格式")
    return header

def read_iq_data(filename, num_samples, sample_format, max_samples=1e6,start_offset=0):
    """
    读取IQ数据（默认读取前100万样本，避免内存不足）
    start_offset:允许从文件的指定字节
    """
    # 限制读取的样本数（根据内存调整）
    read_samples = min(int(max_samples), num_samples)
    total_bytes = read_samples * 2 * bytes_per_sample  # I和Q各占1个样本

    with open(filename, 'rb') as f:
        # 读取二进制数据
        f.seek(start_offset) #移动到分片起始位置
        data = np.fromfile(f, dtype=sample_format, count=2 * read_samples)

    # 分离I和Q分量（交错存储：I0, Q0, I1, Q1, ...）
    I = data[::2].astype(np.float32)  # 偶数索引为I
    Q = data[1::2].astype(np.float32)  # 奇数索引为Q

    # 归一化到[-1, 1]（int16的范围是-32768~32767）
    I /= 32767.0
    Q /= 32767.0

    return I, Q, read_samples
def plot_time_domain(I,Q,Fs):
    t=np.arange(len(I))/Fs

    plt.figure(figsize=(12,8))
    amplitude = np.sqrt(I ** 2 + Q ** 2)
    plt.plot(t, amplitude, color='purple', linewidth=0.5)
    plt.title('Signal Amplitude (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(t[0], t[-1])

    plt.tight_layout()
    plt.show()

#采样点数量=采样率×信号时长  25.2
def detect_and_save_pulses(amplitude, Fs, save_dir="pulses", save_path="pulses_time_domain",
                           threshold_factor=3, min_width=22, max_width=28,
                           max_diff_factor=0.3, max_flatness_std=0.2, min_flatness_min_max=0.5):
    """
    检测时域脉冲并保存对应时域图
    I, Q: IQ分量
    Fs: 采样率
    save_dir: 保存图片的文件夹
    threshold_factor: 阈值因子（均值+factor*std）
    min_width: 脉冲最小宽度（采样点数）
    max_fluctuation: 脉冲内幅度标准差/峰值允许最大值
    """
    # 计算阈值
    noise_mean = np.mean(amplitude)
    noise_std = np.std(amplitude)
    threshold = noise_mean + threshold_factor * noise_std

    # 找出超过阈值的点
    above = amplitude > threshold

    # 检测连续区间
    raw_pulses = []
    in_pulse = False
    start = 0
    for i, val in enumerate(above):
        if val and not in_pulse:
            in_pulse = True
            start = i
        elif not val and in_pulse:
            in_pulse = False
            end = i
            if max_width>=end - start >= min_width:  #限定区间在28，22
                raw_pulses.append((start, end))

    # 如果最后还在脉冲中
    if in_pulse:
        end = len(amplitude)
        if end - start >= min_width :
            raw_pulses.append((start, end))

    # 计算每个脉冲的峰均比和平坦度
    print(f"初始有{len(raw_pulses)}个脉冲")
    pulses_metrics = []  # 存储每个脉冲的指标：(start, end, PAPR, 平坦度1, 平坦度2)
    kept_pulses = []  # 保留：平坦度（标准差）<0.2
    discarded_pulses = []
    idx=1
    for (start, end) in raw_pulses:
        # 提取当前脉冲区间的振幅数据
        pulse_amp = amplitude[start:end]  # 注意：原代码中end是区间终点的下一个索引，切片正确

        # 计算基础统计量
        amp_mean = np.mean(pulse_amp)
        amp_max = np.max(pulse_amp)
        amp_min = np.min(pulse_amp)
        amp_std = np.std(pulse_amp)

        # 避免除以零（振幅恒为0的情况，实际脉冲中通常不会出现）
        if amp_mean < 1e-10:
            papr = 0.0
            flatness1 = 0.0
            flatness2 = 0.0
        else:
            # 计算峰均比
            papr = amp_max / amp_mean
            # 计算两种平坦度
            flatness1 = (amp_max - amp_min) / amp_mean  # 定义1：波动范围与均值比
            flatness_std= amp_std / amp_mean  # 定义2：离散程度与均值比

        # 存储结果
        pulses_metrics.append({
            "start": start,
            "end": end,
            "papr": papr,
            "flatness_max_min": flatness1,  # 基于峰值-谷值的平坦度
            "flatness_std": flatness_std # 基于标准差的平坦度
        })
        # print(f"脉冲 {idx};峰均比（PAPR）：{papr:.2f};平坦度（峰值-谷值）：{flatness1:.2f};平坦度（标准差）：{flatness_std:.2f}")
        idx+=1

        if flatness_std < max_flatness_std and flatness1> min_flatness_min_max :
            kept_pulses.append((start, end))
        else:
            discarded_pulses.append((start, end))



    # # 示例：打印第一个脉冲的指标
    # if pulses_metrics:
    #     print(f"共检测到 {len(pulses_metrics)} 个脉冲，参数如下：\n")
    #     idx=1
    #     for pulse in pulses_metrics: # 从1开始计数
    #         if pulse['flatness_std']<max_flatness_std:
    #             print(f"脉冲 {idx};峰均比（PAPR）：{pulse['papr']:.2f};平坦度（峰值-谷值）：{pulse['flatness_max_min']:.2f};平坦度（标准差）：{pulse['flatness_std']:.2f}")
    #             idx+=1
    # else:
    #     print("未检测到符合条件的脉冲")
    #
    #
    #
    # # 绘制时域图
    print(len(amplitude))
    if len(amplitude)<1e4:
        os.makedirs(save_path, exist_ok=True)
        t = np.arange(len(amplitude)) / Fs
        t=t*1e6
        plt.figure(figsize=(12, 6))

        # 先画非脉冲部分（紫色）
        plt.plot(t, amplitude, color='purple', linewidth=0.5, label="非脉冲")

        # 绘制被舍弃的震荡脉冲（红色）
        for (s, e) in discarded_pulses:
            plt.plot(t[s:e], amplitude[s:e], color='red', linewidth=1.0,
                     label="舍弃（震荡）" if s == discarded_pulses[0][0] else "")

        # 绘制保留的尖峰脉冲（绿色）
        for (s, e) in kept_pulses:
            plt.plot(t[s:e], amplitude[s:e], color='green', linewidth=1.0,
                     label="保留（尖峰）" if s == kept_pulses[0][0] else "")

        # 阈值线
        plt.axhline(threshold, color='red', linestyle='--', linewidth=1.0, label="阈值")

        plt.title("时域脉冲检测结果")
        plt.xlabel("时间 (s)")
        plt.ylabel("幅度")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()

        print(f"已绘制并保存脉冲检测总览图: {save_path}")

    # # 创建保存目录
    # os.makedirs(save_dir, exist_ok=True)
    #
    # # 保存每个脉冲的时域图
    # for idx, (s, e) in enumerate(kept_pulses):
    #     t = np.arange(s, e) / Fs
    #     t*=1e6
    #     amp_seg = amplitude[s:e]
    #
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(t, amp_seg, color='blue', linewidth=0.8)
    #     plt.title(f"Pulse {idx+1} (Samples {s}-{e})")
    #     plt.xlabel("Time (μs)")
    #     plt.ylabel("Amplitude")
    #     plt.grid(True)
    #
    #     filename = os.path.join(save_dir, f"pulse_{idx+1}.png")
    #     plt.savefig(filename, dpi=150)
    #     plt.close()
    #
    #     print(f"已保存脉冲 {idx+1} 的时域图: {filename}")
    return kept_pulses, threshold


def detect_model_ac_responses(kept_pulses, Fs, amplitude,
                              delta_t=1.45,  # 相邻脉冲起始间隔(μs)
                              pulse_duration=0.45,  # 单个脉冲时长(μs)
                              allowed_delta_error=0.1,  # 间隔误差容忍(μs)
                              allowed_duration_error=0.1):  # 脉冲时长误差容忍(μs)
    """
    检测Model A/C回复信号
    新判断逻辑：
    1. 从P0开始，后续脉冲与P0的间隔必须是1.45μs的整数倍（i=1~14）
    2. 第7个脉冲位置（间隔10.15μs=1.45*7）若存在脉冲，则不符合
    3. 必须存在最后一个关键脉冲（间隔20.3μs=1.45*14）
    4. 任何脉冲与P0的间隔超过20.3μs，则不符合
    """
    # 1. 转换脉冲为时间信息(微秒)，并计算与P0的间隔（后续会动态更新）
    pulses_info = []
    for (s, e) in kept_pulses:
        t_start = s / Fs * 1e6  # 起始时间(μs)
        t_end = e / Fs * 1e6  # 结束时间(μs)
        duration = t_end - t_start  # 实际时长(μs)
        pulses_info.append({
            's': s, 'e': e,
            't_start': t_start, 't_end': t_end,
            'duration': duration
        })

    responses = []
    max_interval = 14 * delta_t  # 最大允许间隔：14*1.45=20.3μs
    seventh_interval = 7 * delta_t  # 第7个脉冲的间隔：7*1.45=10.15μs

    # 2. 遍历所有可能的前导脉冲(P0)
    for p0 in pulses_info:
        # 过滤不符合时长的P0
        if not (pulse_duration - allowed_duration_error <= p0['duration'] <= pulse_duration + allowed_duration_error):
            continue

        p0_t_start = p0['t_start']

        # 过滤所有与P0间隔超过max_interval的脉冲（这些脉冲直接不符合）
        valid_pulses = [p for p in pulses_info
                        if (p['t_start'] - p0_t_start) <= (max_interval + allowed_delta_error) and (p['t_start'] - p0_t_start)>0]

        # 3. 检查是否存在最后一个关键脉冲（间隔20.3μs）
        p14_candidates = [p for p in valid_pulses
                          if abs((p['t_start'] - p0_t_start) - max_interval) <= allowed_delta_error
                          and (pulse_duration - allowed_duration_error <= p[
                'duration'] <= pulse_duration + allowed_duration_error)]

        if not p14_candidates:  # 必须存在最后一个关键脉冲
            continue



        # 4. 检查中间脉冲（i=1~13）
        for p14 in p14_candidates:
            # 标记中间脉冲状态
            middle_pulses = []
            valid = True
            dt=p14['t_start']-p0_t_start
            if dt>max_interval+allowed_delta_error:
                valid=False
                break

            for p in valid_pulses:
                actual_interval = p['t_start'] - p0_t_start
                k = (actual_interval / 1.45)%1
                if k > 0.3 and k < 0.7:
                    valid = False
                    break


            # 遍历1~13倍间隔（对应i=1到13）
            for i in range(1, 14):
                expected_interval = i * delta_t
                # 查找该间隔是否存在符合条件的脉冲
                found = False
                for p in valid_pulses:
                    actual_interval = p['t_start'] - p0_t_start
                    # 验证间隔是否为1.45的i倍（允许误差）且时长符合
                    if (abs(actual_interval - expected_interval) <= allowed_delta_error and
                            pulse_duration - allowed_duration_error <= p[
                                'duration'] <= pulse_duration + allowed_duration_error):
                        found = True
                        middle_pulses.append({'idx': i, 'exists': True, 'pulse': p, 'interval': actual_interval})
                        break

                if not found:
                    middle_pulses.append({'idx': i, 'exists': False, 'pulse': None, 'interval': None})

                # 关键检查：第7个脉冲（i=7）若存在则无效
                if i == 7 and found:
                    valid = False
                    break

            # 5. 额外检查：所有存在的脉冲必须是1.45的整数倍间隔（排除已过滤的超范围脉冲）
            # （已通过上面的循环确保，因为只检查了i=1~13的倍数）

            if valid:
                # 记录有效应答
                responses.append({
                    'p0': p0,
                    'p14': p14,
                    'middle_pulses': middle_pulses,
                    'start_time': p0_t_start,
                    'end_time': p14['t_end'],
                    'amplitude': amplitude
                })

    # 6. 去重(避免同一应答被重复检测)
    unique_responses = []
    seen = set()
    for resp in responses:
        # 用P0和P14的起始时间作为唯一标识
        key = (round(resp['p0']['t_start'], 3), round(resp['p14']['t_start'], 3))
        if key not in seen:
            seen.add(key)
            unique_responses.append(resp)

    return unique_responses

def plot_ac_responses(responses, save_dir="model_ac_responses"):
    pulse_duration = 0.45
    """绘制Model A/C应答信号图像"""
    os.makedirs(save_dir, exist_ok=True)
    if not responses:
        print("未检测到Model A/C应答信号")
        return

    for i, resp in enumerate(responses):
        # 提取应答时间范围(扩展1μs便于观察)
        start = resp['start_time'] - 1
        end = resp['end_time'] + 1
        p0 = resp['p0']
        p14 = resp['p14']
        amplitude = resp['amplitude']
        Fs = (p0['e'] - p0['s']) / (p0['duration'] * 1e-6)  # 从脉冲计算采样率

        # 提取绘图用的幅度数据
        s_start = max(0, int((start - 0.5) * 1e-6 * Fs))  # 起始采样点
        s_end = min(len(amplitude), int((end + 0.5) * 1e-6 * Fs))  # 结束采样点
        t_plot = np.arange(s_start, s_end) / Fs * 1e6  # 时间轴(μs)
        amp_plot = amplitude[s_start:s_end]

        plt.figure(figsize=(12, 6))
        plt.plot(t_plot, amp_plot, 'b-', linewidth=0.8, label='信号幅度')

        # 标记前导脉冲(P0)和后导脉冲(P14)
        plt.axvspan(p0['t_start'], p0['t_end'], color='green', alpha=0.3, label='前导脉冲(P0)')
        plt.axvspan(p14['t_start'], p14['t_end'], color='green', alpha=0.3, label='后导脉冲(P14)')

        first_missing = True  # 第一个不存在的数据脉冲
        # 标记中间脉冲(存在的为红色，验证脉冲为黄色虚线)
        for pulse in resp['middle_pulses']:
            idx=pulse['idx']
            t_idx = p0['t_start'] + pulse['idx'] * 1.45  # 该位置预期时间
            t_expected_start = p0['t_start'] + idx * 1.45

            # 计算该脉冲的预期时间区间（起始到结束，宽度为脉冲持续时间）
            t_expected_end = t_expected_start + pulse_duration  # 预期结束时间
            if pulse['idx'] == 7:  # 验证脉冲
                plt.axvline(t_idx, color='yellow', linestyle='--', linewidth=2, label='验证脉冲(不存在)')
            elif pulse['exists']:  # 存在的数据脉冲
                p = pulse['pulse']
                plt.axvspan(p['t_start'], p['t_end'], color='red', alpha=0.3, label='数据脉冲(存在)' if pulse['idx']==1 else "")
            else:  # 不存在的数据脉冲
                plt.axvline(t_idx, color='gray', linestyle=':', linewidth=1, label='数据脉冲(不存在)' if pulse['idx']==1 else "")
                # 2. 红色虚线标记预期区间的左右边界（左侧=起始-0.5*时长，右侧=结束+0.5*时长）
                # 区间宽度为脉冲持续时间，确保覆盖应有的时间范围
                t_left = t_expected_start  # 预期区间左边界（起始时间）
                t_right = t_expected_end   # 预期区间右边界（结束时间）
                plt.axvline(t_left, color='red', linestyle='--', linewidth=1.5,
                           label='脉冲预期区间' if first_missing else "")
                plt.axvline(t_right, color='red', linestyle='--', linewidth=1.5)
                # 更新标记，确保图例只显示一次
                if first_missing:
                    first_missing = False

        plt.xlim(start, end)
        plt.title(f"Model A/C应答 {i+1} ({resp['start_time']:.2f}μs - {resp['end_time']:.2f}μs)")
        plt.xlabel("时间 (μs)")
        plt.ylabel("幅度")
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        # 保存图像
        save_path = os.path.join(save_dir, f"ac_response_{i+1}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"已保存应答 {i+1} 图像: {save_path}")


def detect_model_s_responses(pulses, Fs,amplitude, allowed_error=0.2):
    """
    优化版：检测Model S信号，避免重复遍历已确认信号范围内的脉冲
    参数:
        pulses: 符合0.5微秒脉宽的脉冲列表，每个元素为(start_sample, end_sample)
        Fs: 采样率(Hz)
        allowed_error: 时间间隔误差容忍(微秒)，默认±0.2
    返回:
        检测到的Model S信号列表
    """
    # 1. 脉冲时间转换与排序（同原逻辑）
    pulses_info = []
    for (s, e) in pulses:
        t_start = s / Fs * 1e6  # 起始时间(微秒)
        t_end = e / Fs * 1e6  # 结束时间(微秒)
        pulses_info.append({
            "t_start": t_start,
            "t_end": t_end,
            "start_sample": s,
            "end_sample": e,
            "index": len(pulses_info)  # 记录原始索引，用于定位
        })
    pulses_info.sort(key=lambda x: x["t_start"])  # 按时间排序
    model_s_signals = []
    total_pulses = len(pulses_info)
    current_i = 0  # 当前搜索起始索引（替代原for循环的i）

    min_1us_width = 0.8  # 1微秒脉冲最小宽度
    max_1us_width = 1.2  # 1微秒脉冲最大宽度
    i = 1
    # 2. 用while循环遍历，灵活控制起始位置
    while current_i < total_pulses:
        # 确保后续至少有4个脉冲（i+1到i+3，i+4用于验证固定区结束）
        if current_i + 4 >= total_pulses:
            break  # 剩余脉冲不足，终止搜索

        # 提取当前起始脉冲及后续4个脉冲
        p_i = pulses_info[current_i]
        p_i1 = pulses_info[current_i + 1]
        p_i2 = pulses_info[current_i + 2]
        p_i3 = pulses_info[current_i + 3]
        p_i4 = pulses_info[current_i + 4]
        t_i = p_i["t_start"]

        # 3. 固定区验证（同原逻辑）
        delta_i1 = p_i1["t_start"] - t_i
        delta_i2 = p_i2["t_start"] - t_i
        delta_i3 = p_i3["t_start"] - t_i
        delta_i4 = p_i4["t_start"] - t_i

        # 验证间隔是否符合规则
        if not (
                (1 - allowed_error <= delta_i1 <= 1 + allowed_error) and
                (3.5 - allowed_error <= delta_i2 <= 3.5 + allowed_error) and
                (4.5 - allowed_error <= delta_i3 <= 4.5 + allowed_error) and
                (delta_i4 > 8.0 - allowed_error)
        ):
            # 不符合规则，移动到下一个脉冲继续搜索
            current_i += 1
            continue

        # -------------------------- 新增：检测1毫秒脉冲（目标时间段） --------------------------
        # 目标时间段：i脉冲起始时间 +8μs 到 +10.5μs
        window_start_us = t_i + 7   # 窗口起始时间(微秒)
        window_end_us = t_i + 12  # 窗口结束时间(微秒)

        # 将时间转换为采样点（微秒 -> 采样点：t(μs) * Fs / 1e6）
        window_start_sample = round(window_start_us * Fs / 1e6)
        window_end_sample = round(window_end_us * Fs / 1e6)

        # 确保采样点在有效范围内（避免超出IQ数据长度）
        window_start_sample = max(0, window_start_sample)
        window_end_sample = min(len(amplitude), window_end_sample)  # I和Q长度需一致

        # 截取目标时间段内的IQ数据
        amplitude_window = amplitude [window_start_sample:window_end_sample]


        # 配置1毫秒（1000μs）脉冲的宽度范围（采样点）
        # 1000μs对应的采样点数：1000 * Fs / 1e6 = Fs / 1000
        target_width_us = 1 # 1毫秒
        min_width = int((target_width_us - 0.2) * Fs / 1e6)  # 允许-200μs误差
        max_width = int((target_width_us + 0.2) * Fs / 1e6)  # 允许+200μs误差
        min_width = max(1, min_width)  # 确保最小宽度为正

        # 调用脉冲检测函数，检测窗口内的1毫秒脉冲
        window_1ms_pulses,_ = detect_and_save_pulses(
            amplitude =amplitude_window,
            Fs=Fs,
            min_width=min_width,    # 1毫秒脉冲的最小宽度（采样点）
            max_width=max_width,    # 1毫秒脉冲的最大宽度（采样点）
            save_dir="window_pulses",          # 不需要保存图片时设为None
            save_path="window_pulses",
            threshold_factor=0.35,
            min_flatness_min_max=0.4
        )
        us_pulses_info = []
        for (s, e) in window_1ms_pulses:
            t_start = s / Fs * 1e6  # 起始时间(微秒)
            t_end = e / Fs * 1e6  # 结束时间(微秒)
            us_pulses_info.append({
                "t_start": t_start-1,#之前筛选是从i脉冲开始之后的7微秒开始的
                "t_end": t_end,
                "start_sample": s,
                "end_sample": e,
                "index": len(us_pulses_info)  # 记录原始索引，用于定位
            })
            print(f"中间脉冲的开始时间为{t_start-1}")
        us_pulses_info.sort(key=lambda x: x["t_start"])  # 按时间排序


        # 将窗口内的脉冲采样点转换为原始IQ数据的全局采样点
        # -------------------------- 新增：数据区前3微秒解码 --------------------------
        # 解码初始化：前5位为0（字符串便于后续修改，索引0对应第一位）
        decoded_bits = ["0", "0", "0", "0", "0"]
        # 数据区前3微秒时间范围：t_i+8 ~ t_i+11（前导区8微秒后，共3微秒）
        decode_start = t_i + 8.0
        decode_end = t_i + 10.5

        # 步骤1：处理i+4到i+6脉冲（0.5微秒脉宽），判断时间差对应解码位
        # 确定i+4到i+6的索引范围（防止超出总脉冲数）
        max_candidate_idx = min(current_i + 6, total_pulses - 1)
        candidate_pulses = pulses_info[current_i + 4 : max_candidate_idx + 1]  # i+4到i+6

        for p in candidate_pulses:
            delta = p["t_start"] - t_i  # 脉冲与i的时间差（微秒）
            if delta > decode_end + allowed_error:
                continue  # 超出解码范围，跳过

            # 根据时间差设置对应解码位（允许±0.2微秒误差）
            # 第一位（索引0）：delta≈8微秒
            if 8.0 - allowed_error <= delta <= 8.0 + allowed_error:
                decoded_bits[0] = "1"
            # 第二位（索引1）：delta≈8.5微秒
            elif 8.5 - allowed_error <= delta <= 8.5 + allowed_error:
                decoded_bits[1] = "1"
            # 第三位（索引2）：delta≈9微秒
            elif 9.0 - allowed_error <= delta <= 9.0 + allowed_error:
                decoded_bits[2] = "1"
            # 第四位（索引3）：delta≈9.5微秒
            elif 9.5 - allowed_error <= delta <= 9.5 + allowed_error:
                decoded_bits[3] = "1"
            # 第五位（索引4）：delta≈10微秒
            elif 10.0 - allowed_error <= delta <= 10.0 + allowed_error:
                decoded_bits[4] = "1"

        for p in us_pulses_info:
            delta=p['t_start']
            if delta > decode_end + allowed_error:
                continue  # 超出解码范围，跳过

            # 第一位（索引0）：delta≈8微秒
            if 0.0 - allowed_error <= delta <= 0.0 + allowed_error:
                decoded_bits[0] = "1"
                decoded_bits[1] = "1"
            # 第二位（索引1）：delta≈8.5微秒
            elif 0.5 - allowed_error <= delta <= 0.5 + allowed_error:
                decoded_bits[1] = "1"
                decoded_bits[2] = "1"
            # 第三位（索引2）：delta≈9微秒
            elif 1.0 - allowed_error <= delta <= 1.0 + allowed_error:
                decoded_bits[2] = "1"
                decoded_bits[3] = "1"
            # 第四位（索引3）：delta≈9.5微秒
            elif 1.5 - allowed_error <= delta <= 1.5 + allowed_error:
                decoded_bits[3] = "1"
                decoded_bits[4] = "1"
            # 第五位（索引4）：delta≈10微秒
            elif 2.0 - allowed_error <= delta <= 2.0 + allowed_error:
                decoded_bits[4] = "1"


        decoded_result="".join(decoded_bits)
        if decoded_result=="10001":
            type="ADS-B"
        else:
            type="others"
        # 4. 数据区检测（找到该信号包含的最远脉冲索引）
        max_delta = 0.0
        farthest_n = current_i  # 记录最远脉冲的索引
        for n in range(current_i, total_pulses):
            p_n = pulses_info[n]
            delta = p_n["t_start"] - t_i
            if delta > 120.0 + allowed_error:
                break  # 超出最大总长度，停止检查
            if delta > max_delta:
                max_delta = delta
                farthest_n = n  # 更新最远脉冲索引

        # 5. 判断数据区长度
        if max_delta <= 64.0 + allowed_error:
            data_length = 56.0
            total_length = 64.0
        elif max_delta <= 120.0 + allowed_error:
            data_length = 112.0
            total_length = 120.0
        else:
            # 不符合长度规则，移动到下一个脉冲
            current_i += 1
            continue

        # 6. 记录有效信号
        model_s_signals.append({
            "start_time": t_i,
            "end_time": t_i + total_length,
            "data_length": data_length,
            "total_length": total_length,
            "start_pulse_idx": current_i,
            "farthest_pulse_idx": farthest_n,  # 记录最远脉冲索引
            "max_pulse_delta": max_delta,
            "decoded_bits":decoded_result, # 新增解码结果前5位
            "type":type
        })

        # 7. 关键优化：下一次搜索从最远脉冲的下一个开始，跳过已处理范围
        current_i = farthest_n + 1

    # 8. 去重（同原逻辑）
    unique_signals = []
    seen = set()
    for sig in model_s_signals:
        key = (round(sig["start_time"], 2), round(sig["total_length"], 1))
        if key not in seen:
            seen.add(key)
            unique_signals.append(sig)

    return unique_signals


def plot_model_s_responses(signals, pulses, amplitude, Fs, save_dir="model_s_plots"):
    """
    绘制Model S回复信号的时域图像并保存
    参数:
        signals: 从detect_model_s_responses返回的Model S信号列表
        pulses: 符合0.5微秒脉宽的脉冲列表（(start_sample, end_sample)）
        amplitude: 原始信号幅度数据（时域）
        Fs: 采样率(Hz)
        save_dir: 图像保存文件夹
    """
    # 创建保存文件夹
    os.makedirs(save_dir, exist_ok=True)

    if not signals:
        print("未检测到Model S信号，无需绘图")
        return

    # 1. 预处理脉冲时间信息（转换采样点到微秒）
    pulses_info = []
    for (s, e) in pulses:
        t_start = s / Fs * 1e6  # 脉冲起始时间（微秒）
        t_end = e / Fs * 1e6  # 脉冲结束时间（微秒）
        pulses_info.append({
            "t_start": t_start,
            "t_end": t_end,
            "start_sample": s,
            "end_sample": e
        })
    pulses_info.sort(key=lambda x: x["t_start"])  # 按时间排序

    # 2. 遍历每个信号绘图
    for idx, sig in enumerate(signals, 1):
        # 提取信号关键参数
        start_time = sig["start_time"]  # 信号起始时间（微秒）
        end_time = sig["end_time"]  # 信号结束时间（微秒）
        decoded_result = sig["decoded_bits"]
        type=sig["type"]
        data_length = sig["data_length"]  # 数据区长度（微秒）
        start_pulse_idx = sig["start_pulse_idx"]  # 起始脉冲在pulses_info中的索引
        farthest_pulse_idx = sig["farthest_pulse_idx"]  # 最远脉冲索引

        # 3. 计算绘图时间范围（扩展10%边界，便于观察）
        time_pad = (end_time - start_time) * 0.1
        plot_start = start_time - time_pad
        plot_end = end_time + time_pad

        # 4. 转换时间到采样点（用于提取幅度数据）
        sample_to_us = 1e6 / Fs  # 1采样点对应的微秒数
        start_sample = int(plot_start / sample_to_us)
        end_sample = int(plot_end / sample_to_us)
        # 防止索引越界
        start_sample = max(0, start_sample)
        end_sample = min(len(amplitude), end_sample)

        # 5. 提取绘图用的时间轴和幅度数据
        t = np.arange(start_sample, end_sample) * sample_to_us  # 时间轴（微秒）
        amp_segment = amplitude[start_sample:end_sample]  # 幅度数据

        # 6. 创建图像
        plt.figure(figsize=(12, 6))

        # 绘制时域幅度波形
        plt.plot(t, amp_segment, color='navy', linewidth=0.8, label='信号幅度')

        # 7. 标记固定区（8微秒）和数据区
        # 固定区背景（浅绿色）
        plt.axvspan(start_time, start_time + 8,
                    color='lightgreen', alpha=0.3,
                    label=f'固定区（8μs）')
        # 数据区背景（浅黄色）
        plt.axvspan(start_time + 8, end_time,
                    color='lightyellow', alpha=0.3,
                    label=f'数据区（{data_length}μs）')

        # 8. 标记信号内的所有脉冲
        # 提取当前信号包含的脉冲
        signal_pulses = pulses_info[start_pulse_idx: farthest_pulse_idx + 1]
        for p in signal_pulses:
            # 区分固定区和数据区脉冲
            if p["t_start"] < start_time + 7:
                # 固定区脉冲（红色）
                plt.axvspan(p["t_start"], p["t_end"],
                            color='red', alpha=0.5,
                            label='固定区脉冲' if p == signal_pulses[0] else "")
            else:
                # 数据区脉冲（橙色）
                plt.axvspan(p["t_start"], p["t_end"],
                            color='orange', alpha=0.5,
                            label='数据区脉冲' if p == signal_pulses[4] else "")

        # 9. 标记固定区关键脉冲（i, i+1, i+2, i+3）及间隔
        key_pulses = pulses_info[start_pulse_idx: start_pulse_idx + 4]
        for k, p in enumerate(key_pulses):
            # 脉冲位置标记（紫色圆点）
            plt.scatter(p["t_start"], np.max(amp_segment) * 0.9,
                        color='purple', s=50, zorder=5,
                        label=f'关键脉冲（i+{k}）' if k == 0 else "")
            # 标注时间间隔
            if k > 0:
                delta = p["t_start"] - key_pulses[0]["t_start"]
                plt.text(
                    (key_pulses[0]["t_start"] + p["t_start"]) / 2,
                    np.max(amp_segment) * 0.85,
                    f'{delta:.1f}μs',
                    color='purple',
                    fontsize=9,
                    ha='center'
                )
        plt.text(
            start_time + (end_time - start_time) * 0.05,  # 水平位置：信号左侧5%
            np.max(amp_segment) * 0.9,  # 垂直位置：幅度最大值90%处
            f"解码结果（前5位）：{decoded_result},信号类型{type}",
            color="darkred",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightpink", alpha=0.7)
        )
        # 10. 图像美化
        plt.xlim(plot_start, plot_end)
        plt.ylim(0, np.max(amp_segment) * 1.1)  # 幅度轴留10%余量
        plt.title(f'Model S回复信号 {idx}（总长度 {sig["total_length"]}μs）', fontsize=12)
        plt.xlabel('时间（μs）', fontsize=10)
        plt.ylabel('幅度', fontsize=10)
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # 11. 保存图像
        save_path = os.path.join(save_dir, f'model_s_signal_{idx}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()  # 释放内存
        print(f"已保存Model S信号图像：{save_path}")
def main(iqh_path,iq_path):
    # 1. 读取头文件
    header = read_iqh_header(iqh_path)
    sample_rate = header['SampleRate']
    data_scale = header['DataScale']
    num_sample_total=header['NumberSamples']
    max_samples_per_chunk = 1e6  # 每个分片的最大样本数（保留限制）
    max_samples_per_chunk = int(max_samples_per_chunk)
    total_chunks = int(np.ceil(num_sample_total / max_samples_per_chunk))  # 总分片数
    print(f"成功读取头文件：采样率={sample_rate / 1e6:.1f}MHz，总样本数={header['NumberSamples']}，分{total_chunks}个分片处理（每个分片最多{max_samples_per_chunk}样本）")
    start_offset=0
    # 2. 读取IQ数据并转换为幅度
    I, Q, read_samples = read_iq_data(
        iq_path,
        num_sample_total,
        sample_format,
        max_samples=5e6,# 读取500万样本（约40MB）
        start_offset=int(start_offset)
    )
    amplitude = np.sqrt(I ** 2 + Q ** 2)
    #采样点数量=采样率×信号时长  25.2
    ac_width=sample_rate * 4.5e-7

    ac_pulses, threshold = detect_and_save_pulses(amplitude, sample_rate,
                                               save_dir=iq_path+"detected_pulses",
                                               threshold_factor=1,
                                               min_width=0.85*ac_width,max_width=1.15*ac_width)
    print(f"检测到{len(ac_pulses)}个0.45微秒脉宽信号")

    # 第二步：检测Model A/C应答信号
    ac_responses = detect_model_ac_responses(ac_pulses, sample_rate, amplitude)
    if ac_responses:
        print(f"\n共检测到 {len(ac_responses)} 个Model A/C应答信号")
        for i, resp in enumerate(ac_responses):
            print(f"应答 {i+1}: {resp['start_time']:.2f}μs - {resp['end_time']:.2f}μs")

        # 第三步：绘制应答信号图像
        # plot_ac_responses(ac_responses)
    else:
        print("\n未检测到Model A/C应答信号")

#   采样点数=采样率+信号时长  28
    s_width=sample_rate * 5e-7
    s_pulses, threshold = detect_and_save_pulses(amplitude, sample_rate,
                                               save_dir=iq_path+"detected_s_pulses",
                                               threshold_factor=2,
                                               min_width=0.7*s_width,max_width=1.3*s_width,
                                                 max_flatness_std=0.25, min_flatness_min_max=0.45)
    print(f"共有{len(s_pulses)}个0.5微秒脉宽信号")

    s_responses = detect_model_s_responses(amplitude=amplitude,
        pulses=s_pulses,  # 从detect_and_save_pulses得到的0.5微秒脉宽脉冲
        Fs=sample_rate,
        allowed_error=0.2  # 时间间隔误差容忍±0.2微秒
    )
    if s_responses:
        print(f"\n共检测到{len(s_responses)}个Model S回复信号，开始绘图...")
        for idx, sig in enumerate(s_responses, 1):
            print(f"信号{idx}：起始时间：{sig['start_time']:.2f}微秒,数据区长度：{sig['data_length']}微秒")
        plot_model_s_responses(
            signals=s_responses,
            pulses=s_pulses,
            amplitude=amplitude,  # 原始幅度数据
            Fs=sample_rate,
            save_dir="model_s_responses_plots"  # 保存文件夹名称
        )
    else:
        print("\n未检测到Model S回复信号，不绘图")





if __name__ == "__main__":
    # 替换为你的实际文件路径（相对路径/绝对路径均可）
    IQH_PATH = "data/TCAS-40m_1.09G_40M_20251001_174929.391.iqh"
    IQ_PATH = "data/TCAS-40m_1.09G_40M_20251001_174929.391.iqb"

    # 检查文件是否存在
    if not os.path.exists(IQH_PATH):
        raise FileNotFoundError(f"找不到IQ头文件：{os.path.abspath(IQH_PATH)}")
    if not os.path.exists(IQ_PATH):
        raise FileNotFoundError(f"找不到IQ数据文件：{os.path.abspath(IQ_PATH)}")

    # 启动主程序
    main(IQH_PATH, IQ_PATH)