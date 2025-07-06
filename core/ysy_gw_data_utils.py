import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import ysy_plot_utils as ypu
from scipy import signal

def get_PSD(data, fs, duration):
    num_samples = int(fs * duration)
    freqs, psd = welch(data, fs=fs, nperseg=num_samples, noverlap=num_samples//2, window='blackman')
    return freqs, psd

def whiten(data, psd, freqs, dt):
    fill_value = psd[-1]
    norm = 1./np.sqrt(1./(dt*2))
    psd_func = interp1d(freqs, psd, bounds_error=False, fill_value=fill_value)
    ft_data = np.fft.rfft(data)
    Nt = len(data)
    ft_freqs = np.fft.rfftfreq(Nt, dt)
    whiten_ft_data = ft_data * norm / np.sqrt(psd_func(ft_freqs))
    whiten_data = np.fft.irfft(whiten_ft_data, n=Nt)
    return whiten_data

def band_filter(data, fband, fs):
    bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
    normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
    selected_data = filtfilt(bb, ab, data)/ normalization
    return selected_data

class PreProcessing:
    def __init__(self, event_name ,fs, t_event, fband, H1, L1):
        self.event_name = event_name
        self.fs = fs
        self.t_event = t_event
        self.fband = fband
        self.H1 = H1
        self.L1 =L1
    
    def main(self):
        data_ready = {}

        event_name = self.event_name
        fs = self.fs
        t_event = self.t_event
        fband = self.fband
        H1 = self.H1
        L1 =self.L1

        H1_data = pd.read_csv(H1)
        L1_data = pd.read_csv(L1)
        strain_H1 = H1_data['strain'].values
        time_H1 = H1_data['GPS_time'].values
        strain_L1 = L1_data['strain'].values
        time_L1 = L1_data['GPS_time'].values
        data_ready.update({
            'strain_H1': strain_H1,
            'time_H1': time_H1, 
            'strain_L1': strain_L1, 
            'time_L1': time_L1, 
        })

        time = time_H1
        dt = time[1] - time[0]
        duration = 4

        f_H1, psd_H1 = get_PSD(strain_H1, fs, duration)
        f_L1, psd_L1 = get_PSD(strain_L1, fs, duration)
        data_ready.update({
            'f_H1': f_H1,
            'f_L1': f_L1, 
            'psd_H1': psd_H1, 
            'psd_L1': psd_L1, 
        })

        whiten_H1 = whiten(strain_H1, psd_H1, f_H1, dt)
        whiten_L1 = whiten(strain_L1, psd_L1, f_L1, dt)
        data_ready.update({
            'whiten_H1': whiten_H1,
            'whiten_L1': whiten_L1, 
        })

        selected_whiten_H1 = band_filter(whiten_H1, fband, fs)
        selected_whiten_L1 = band_filter(whiten_L1, fband, fs)
        data_ready.update({
            'selected_whiten_H1': selected_whiten_H1,
            'selected_whiten_L1': selected_whiten_L1, 
        })

        return data_ready
    
def process_and_plot_event(event_data):
    """
    处理单个引力波事件数据并生成三张图。

    参数:
    event_data (dict): 包含单个事件信息的字典。
    """
    # --- 1. 从字典中提取事件的基本信息 ---
    event_name = event_data['name']
    fs = event_data['fs']
    t_event = event_data['tevent']
    fband = event_data['fband']
    H1 = event_data['fn_H1']
    L1 = event_data['fn_L1']
    
    print(f"--- 开始处理事件: {event_name} ---")

    # --- 2. 使用 ygdu.PreProcessing 进行数据预处理 ---
    # 假设 PreProcessing 类需要这些参数进行初始化
    event_processor = PreProcessing(event_name, fs, t_event, fband, H1, L1)
    processed_data = event_processor.main()

    # --- 3. 提取处理后的数据 ---
    strain_H1 = processed_data['strain_H1']
    time_H1 = processed_data['time_H1']
    strain_L1 = processed_data['strain_L1']
    
    selected_whiten_H1 = processed_data['selected_whiten_H1']
    selected_whiten_L1 = processed_data['selected_whiten_L1']

    # 假设 H1 和 L1 的时间轴是相同的
    time = time_H1

    # --- 4. 绘图 ---

    # 图 1: 绘制事件发生前后一小段时间内的原始应变数据
    delta_t1 = 5
    idx1 = np.where((time >= t_event - delta_t1) & (time < t_event + delta_t1))
    
    with ypu.temp_style(["ysy_academic", "science_color"]):
        ypu.plot(
            time[idx1] - t_event, 
            (strain_H1[idx1], strain_L1[idx1]), 
            ('H1', 'L1'), 
            f'Strain Data near {event_name}', 
            f'Time (since {t_event}s) [s]', 
            'Strain'
        )

    # 图 2: 绘制事件发生时刻附近极短时间内的白化应变数据
    delta_t2 = 0.1
    delta_t3 = 0.05
    idx2 = np.where((time >= t_event - delta_t2) & (time < t_event + delta_t3))

    with ypu.temp_style(["ysy_academic", "science_color"]):
        ypu.plot(
            time[idx2] - t_event, 
            (selected_whiten_H1[idx2], selected_whiten_L1[idx2]), 
            ('H1', 'L1'), 
            f'Selected Whiten Strain Data near {event_name}', 
            f'Time (since {t_event}s) [s]', 
            'Strain'
        )

    # 图 3: 在图1相同的时间范围内绘制白化应变数据以作对比
    with ypu.temp_style(["ysy_academic", "science_color"]):
        ypu.plot(
            time[idx1] - t_event, 
            (selected_whiten_H1[idx1], selected_whiten_L1[idx1]), 
            ('H1', 'L1'), 
            f'Selected Whiten Strain Data near {event_name} (Wide View)', 
            f'Time (since {t_event}s) [s]', 
            'Strain'
        )
    
    print(f"--- 事件 {event_name} 处理完毕 ---\n")
    return None

def load_template(template_name, plot=False):
    template_data = pd.read_csv(template_name)
    template_p = template_data.iloc[0].values
    template_c = template_data.iloc[1].values
    template = template_p + 1j * template_c

    if plot:
        x = np.linspace(0, len(template_c), len(template_c))
        ypu.plot(x, template_p, 'Wave Template P')
        ypu.plot(x, template_c, 'Wave Template C')
        ypu.plot(template_p, template_c, 'Wave Template P-C')

    return template

def prepare_whitened_template(template, psd, freqs, fs, dt, need_ads=False):
    """
    参数：
    psd: array, 探测器的功率谱密度
    freqs: array, 与PSD对应的频率数组

    返回：
    template_fft_whitened: array, 白化后的频率域复数模板
    """
    N = len(template)
    tukey_a = 0.5
    try:
        dwindow = signal.windows.tukey(N, alpha=tukey_a)
    except AttributeError:
        dwindow = signal.windows.blackman(template.size)
    template_fft = np.fft.fft(template*dwindow) / fs
    
    fft_freqs = np.fft.fftfreq(N, d=dt)
    try:
        power_vec = np.interp(np.abs(fft_freqs), freqs, psd)
    except Exception as e:
        print(f"PSD插值时发生错误: {e}")
        print("请检查 freqs 和 psd 数组是否有效。")
        return None
    asd_vec = np.sqrt(power_vec)
    asd_vec[asd_vec == 0] = np.inf
    
    template_fft_whitened = template_fft / asd_vec
    template_fft_whitened[fft_freqs == 0] = 0

    if need_ads:
        return template_fft_whitened, asd_vec
    else:
        return template_fft_whitened
        
def calculate_snr_from_whitened(data_whitened, template_fft_whitened, fs, dt):
    """
    参数:
    data_whitened: array, 时间域的白化探测器数据
    template_fft_whitened: array, 频率域的白化复数模板

    返回:
    snr: array, SNR时间序列
    phase  
    offset  
    d_eff  
    """
    N = len(data_whitened)
    df = fs / N
    tukey_a = 0.5
    try:
        dwindow = signal.windows.tukey(N, alpha=tukey_a)
    except AttributeError:
        dwindow = signal.windows.blackman(N) 
    
    norm = 1./np.sqrt(1./(dt*2))
    data_fft_whitened = np.fft.fft(data_whitened*dwindow) / (fs*norm)
    
    sigmasq = np.sum(template_fft_whitened*template_fft_whitened.conj()) * df
    sigma = np.sqrt(np.abs(sigmasq))
    
    mf_fft = data_fft_whitened * template_fft_whitened.conj()
    
    mf_time = 2*np.fft.ifft(mf_fft) * fs
    
    snr_complex = mf_time / sigma
    
    peaksample = N // 2
    snr_shifted = np.roll(snr_complex, shift=peaksample)

    snr = np.abs(snr_shifted)
    indmax = np.argmax(snr)
    phase = np.angle(snr_shifted[indmax])
    offset = (indmax-peaksample)
    peak_snr = np.max(snr)
    d_eff = sigma / peak_snr
    
    return snr, phase, offset, d_eff

def calculate_snr_from_selected_whiten(selected_whiten_data, template_fft_whitened, fs, dt):
    """
    参数:
    data_whitened: array, 时间域的白化探测器数据
    template_fft_whitened: array, 频率域的白化复数模板

    返回:
    snr: array, SNR时间序列
    phase  
    offset  
    d_eff  
    """
    N = len(selected_whiten_data)
    df = fs / N
    tukey_a = 0.5
    try:
        dwindow = signal.windows.tukey(N, alpha=tukey_a)
    except AttributeError:
        dwindow = signal.windows.blackman(N) 
    
    data_fft_whitened = np.fft.fft(selected_whiten_data*dwindow) / fs
    
    sigmasq = np.sum(template_fft_whitened*template_fft_whitened.conj()) * df
    sigma = np.sqrt(np.abs(sigmasq))
    
    mf_fft = data_fft_whitened * template_fft_whitened.conj()
    
    mf_time = 2*np.fft.ifft(mf_fft) * fs
    
    snr_complex = mf_time / sigma
    
    peaksample = N // 2
    snr_shifted = np.roll(snr_complex, shift=peaksample)

    snr = np.abs(snr_shifted)
    indmax = np.argmax(snr)
    phase = np.angle(snr_shifted[indmax])
    offset = (indmax-peaksample)
    peak_snr = np.max(snr)
    d_eff = sigma / peak_snr
    
    return snr, phase, offset, d_eff

def matched_filtering_main(event_data, band_pass=False):

    event_name = event_data['name']
    fs = event_data['fs']
    t_event = event_data['tevent']
    fband = event_data['fband']
    H1 = event_data['fn_H1']
    L1 = event_data['fn_L1']
    template_name = event_data['fn_template']

    event_processor = PreProcessing(event_name, fs, t_event, fband, H1, L1)
    processed_data = event_processor.main()
    template = load_template(template_name)
    
    time_H1 = processed_data['time_H1']
    selected_whiten_H1 = processed_data['selected_whiten_H1']
    selected_whiten_L1 = processed_data['selected_whiten_L1']
    time = time_H1
    dt = time[1] - time[0]
    psd_H1 = processed_data['psd_H1']
    f_H1 = processed_data['f_H1']
    psd_L1 = processed_data['psd_L1']
    f_L1 = processed_data['f_L1']

    whiten_ft_template_H1 = prepare_whitened_template(template, psd_H1, f_H1, fs, dt)
    whiten_ft_template_L1 = prepare_whitened_template(template, psd_L1, f_L1, fs, dt)

    whiten_H1 = processed_data['whiten_H1']
    if band_pass:
        snr_H1, phase_H1, offset_H1, d_eff_H1 = calculate_snr_from_selected_whiten(selected_whiten_H1, whiten_ft_template_H1, fs, dt)
    else:
        snr_H1, phase_H1, offset_H1, d_eff_H1 = calculate_snr_from_whitened(whiten_H1, whiten_ft_template_H1, fs, dt)
    peak_snr_H1 = np.max(snr_H1)
    peak_index_H1 = np.argmax(snr_H1)
    peak_time_H1 = time_H1[peak_index_H1]
    print(event_name+" H1: We find max SNR", peak_snr_H1, 'at', peak_time_H1, 's with the effctive distance', d_eff_H1, '.')

    template_phase_shifted_H1 = np.real(template*np.exp(1j*phase_H1))
    template_rolled_H1 = np.roll(template_phase_shifted_H1, offset_H1) / d_eff_H1
    whiten_template_H1 = whiten(template_rolled_H1, psd_H1, f_H1, dt)
    selected_whiten_template_H1 = band_filter(whiten_template_H1, fband, fs)

    delta_t2 = 0.1
    delta_t3 = 0.05
    indxt2 = np.where((time >= t_event-delta_t2) & (time < t_event+delta_t3))

    with ypu.temp_style(["ysy_academic", "science_color"]):
        ypu.plot(
            time-peak_time_H1, snr_H1, 'H1 SNR(t)', 
            'H1 SNR near '+event_name, 'Time (since {0:.4f}s) [s]'.format(peak_time_H1), 'SNR'
        )

        ypu.plot(
            time[indxt2]-t_event, (selected_whiten_H1[indxt2], selected_whiten_template_H1[indxt2]), ('H1', 'Wave Template'), 
            'Selected Whiten H1 with Matched filter \nnear '+event_name, 'Time (since '+str(t_event)+'s) [s]', 'Strain'
        )
    
    whiten_L1 = processed_data['whiten_L1']
    if band_pass:
        snr_L1, phase_L1, offset_L1, d_eff_L1 = calculate_snr_from_whitened(selected_whiten_L1, whiten_ft_template_L1, fs, dt)
    else:
        snr_L1, phase_L1, offset_L1, d_eff_L1 = calculate_snr_from_whitened(whiten_L1, whiten_ft_template_L1, fs, dt)

    # --- 寻找并输出峰值结果 ---
    peak_snr_L1 = np.max(snr_L1)
    peak_index_L1 = np.argmax(snr_L1)
    peak_time_L1 = time_H1[peak_index_L1]
    print(event_name+" L1: We find max SNR", peak_snr_L1, 'at', peak_time_L1, 's with the effctive distance', d_eff_L1, '.')
    
    template_phase_shifted_L1 = np.real(template*np.exp(1j*phase_L1))
    template_rolled_L1 = np.roll(template_phase_shifted_L1, offset_L1) / d_eff_L1
    whiten_template_L1 = whiten(template_rolled_L1, psd_L1, f_L1, dt)
    selected_whiten_template_L1 = band_filter(whiten_template_L1, fband, fs)

    with ypu.temp_style(["ysy_academic", "science_color"]):
        ypu.plot(
            time-peak_time_L1, snr_L1, 'L1 SNR(t)', 
            'L1 SNR near '+event_name, 'Time (since {0:.4f}s) [s]'.format(peak_time_L1), 'SNR'
        )

        ypu.plot(
            time[indxt2]-t_event, (selected_whiten_L1[indxt2], selected_whiten_template_L1[indxt2]), ('L1', 'Wave Template'), 
            'Selected Whiten L1 with Matched filter \nnear '+event_name, 'Time (since '+str(t_event)+'s) [s]', 'Strain'
        )
    return None

def matched_filtering_main_2(event_info, plot=True):
    result = {}

    event_name = event_info['name']
    fs = event_info['fs']
    t_event = event_info['tevent']
    fband = event_info['fband']
    H1 = event_info['fn_H1']
    L1 = event_info['fn_L1']
    template_name = event_info['fn_template']

    template = load_template(template_name)
    datafreq = np.fft.fftfreq(template.size)*fs
    df = np.abs(datafreq[1] - datafreq[0])
    NFFT = 4*fs
    psd_window = np.blackman(NFFT)
    NOVL = NFFT // 2
    try:   dwindow = signal.windows.tukey(template.size, alpha=1./8)
    except: dwindow = signal.windows.blackman(template.size)
    template_fft = np.fft.fft(template*dwindow) / fs

    pre_progress = PreProcessing(event_name, fs, t_event, fband, H1, L1)
    data = pre_progress.main()

    strain_H1 = data['strain_H1']
    strain_L1 = data['strain_L1']
    time = data['time_H1']
    dt = time[1] - time[0]
    f_H1 = data['f_H1']
    psd_H1 = data['psd_H1']
    f_L1 = data['f_L1']
    psd_L1 = data['psd_L1']
    selected_whiten_H1 = data['selected_whiten_H1']
    selected_whiten_L1 = data['selected_whiten_L1']

    H1_fft = np.fft.fft(strain_H1*dwindow) / fs
    L1_fft = np.fft.fft(strain_L1*dwindow) / fs

    power_vec_H1 = np.interp(np.abs(datafreq), f_H1, psd_H1)
    power_vec_L1 = np.interp(np.abs(datafreq), f_L1, psd_L1)

    optimal_H1 = H1_fft * template_fft.conjugate() / power_vec_H1
    optimal_time_H1 = 2*np.fft.ifft(optimal_H1)*fs

    optimal_L1 = L1_fft * template_fft.conjugate() / power_vec_L1
    optimal_time_L1 = 2*np.fft.ifft(optimal_L1)*fs

    sigmasq_H1 = 1*(template_fft * template_fft.conjugate() / power_vec_H1).sum() * df
    sigma_H1 = np.sqrt(np.abs(sigmasq_H1))
    SNR_complex_H1 = optimal_time_H1/sigma_H1

    sigmasq_L1 = 1*(template_fft * template_fft.conjugate() / power_vec_L1).sum() * df
    sigma_L1 = np.sqrt(np.abs(sigmasq_L1))
    SNR_complex_L1 = optimal_time_L1/sigma_L1

    peaksample_H1 = int(strain_H1.size / 2)
    SNR_complex_H1 = np.roll(SNR_complex_H1,peaksample_H1)
    SNR_H1 = abs(SNR_complex_H1)

    peaksample_L1 = int(strain_L1.size / 2)
    SNR_complex_L1 = np.roll(SNR_complex_L1,peaksample_L1)
    SNR_L1 = abs(SNR_complex_L1)

    indmax_H1 = np.argmax(SNR_H1)
    timemax_H1 = time[indmax_H1]
    SNRmax_H1 = SNR_H1[indmax_H1]

    indmax_L1 = np.argmax(SNR_L1)
    timemax_L1 = time[indmax_L1]
    SNRmax_L1 = SNR_L1[indmax_L1]

    d_eff_H1 = sigma_H1 / SNRmax_H1
    horizon_H1 = sigma_H1/8
    phase_H1 = np.angle(SNR_complex_H1[indmax_H1])
    offset_H1 = (indmax_H1-peaksample_H1)

    d_eff_L1 = sigma_L1 / SNRmax_L1
    horizon_L1 = sigma_L1/8
    phase_L1 = np.angle(SNR_complex_L1[indmax_L1])
    offset_L1 = (indmax_L1-peaksample_L1)

    result.update(
        {
            'timemasx_H1': timemax_H1, 
            'SNRmax_H1': SNRmax_H1, 
            'phase_H1': phase_H1,
            'd_eff_H1': d_eff_H1, 
            'timemasx_L1': timemax_L1, 
            'SNRmax_L1': SNRmax_L1, 
            'phase_L1': phase_L1,
            'd_eff_L1': d_eff_L1, 
        }
    )

    delta_t2 = 0.1
    delta_t3 = 0.05
    indxt2 = np.where((time >= t_event-delta_t2) & (time < t_event+delta_t3))

    template_phaseshifted_H1 = np.real(template*np.exp(1j*phase_H1))
    template_rolled_H1 = np.roll(template_phaseshifted_H1,offset_H1) / d_eff_H1
    template_whitened_H1 = whiten(template_rolled_H1, psd_H1, f_H1, dt)
    template_match_H1 = band_filter(template_whitened_H1, fband, fs)
    print(
        event_name+': For detector H1, maximum at {0:.4f} with SNR = {1:.1f}, D_eff = {2:.2f}, horizon = {3:0.1f} Mpc'
        .format(timemax_H1,SNRmax_H1,d_eff_H1,horizon_H1)
    )
    if plot:
        with ypu.temp_style(["ysy_academic", "science_color"]):
            ypu.plot(
                time-timemax_H1, SNR_H1, 'H1 SNR(t)', 
                'H1 SNR near '+event_name, 'Time (since {0:.4f}s) [s]'.format(timemax_H1), 'SNR'
            )
            ypu.plot(
                time[indxt2]-t_event, (selected_whiten_H1[indxt2], template_match_H1[indxt2]), ('H1', 'Wave Template'), 
                'Selected Whiten H1 with Matched filter \nnear '+event_name, 'Time (since '+str(t_event)+'s) [s]', 'Strain'
            )
    
    template_phaseshifted_L1 = np.real(template*np.exp(1j*phase_L1))
    template_rolled_L1 = np.roll(template_phaseshifted_L1,offset_L1) / d_eff_L1
    template_whitened_L1 = whiten(template_rolled_L1, psd_L1, f_L1, dt)
    template_match_L1 = band_filter(template_whitened_L1, fband, fs)
    print(
        event_name+': For detector L1, maximum at {0:.4f} with SNR = {1:.1f}, D_eff = {2:.2f}, horizon = {3:0.1f} Mpc'
        .format(timemax_L1,SNRmax_L1,d_eff_L1,horizon_L1)
    )
    if plot:
        with ypu.temp_style(["ysy_academic", "science_color"]):
            ypu.plot(
                time-timemax_L1, SNR_L1, 'L1 SNR(t)', 
                'L1 SNR near '+event_name, 'Time (since {0:.4f}s) [s]'.format(timemax_L1), 'SNR'
            )
            ypu.plot(
                time[indxt2]-t_event, (selected_whiten_L1[indxt2], template_match_L1[indxt2]), ('L1', 'Wave Template'), 
                'Selected Whiten L1 with Matched filter \nnear '+event_name, 'Time (since '+str(t_event)+'s) [s]', 'Strain'
            )
    
    return result

def caculate_coherent_snr(event_info):
    event_name = event_info['name']
    fs = event_info['fs']
    t_event = event_info['tevent']
    fband = event_info['fband']
    H1 = event_info['fn_H1']
    L1 = event_info['fn_L1']
    template_name = event_info['fn_template']

    template = load_template(template_name)
    datafreq = np.fft.fftfreq(template.size)*fs
    df = np.abs(datafreq[1] - datafreq[0])
    try:   dwindow = signal.windows.tukey(template.size, alpha=1./8)
    except: dwindow = signal.windows.blackman(template.size)
    template_fft = np.fft.fft(template*dwindow) / fs

    pre_progress = PreProcessing(event_name, fs, t_event, fband, H1, L1)
    data = pre_progress.main()

    strain_H1 = data['strain_H1']
    strain_L1 = data['strain_L1']
    strain = strain_H1 + strain_L1
    f, psd = get_PSD(strain, fs, 4)
    time = data['time_H1']
    dt = time[1] - time[0]

    strain_fft = np.fft.fft(strain*dwindow) / fs

    power_vec = np.interp(np.abs(datafreq), f, psd)
    
    optimal = strain_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)*fs

    sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    peaksample = int(strain.size / 2)
    SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)

    indmax = np.argmax(SNR)
    timemax = time[indmax]
    SNRmax = SNR[indmax]

    return SNRmax
    
def calculate_time_score(t_h1: float, t_l1: float, k_penalty: float = 10.0) -> float:
    """
    计算到达时间一致性得分 (S_time)。

    得分基于测量的时间差与物理上可能的最大时间差的比较。
    如果时间差在物理允许范围内，得分为1。如果超出，得分会受到指数惩罚。

    公式:
    S_time = exp( -k * ( (max(0, dt - dt_max) / dt_max)^2 ) )

    参数:
    t_h1 (float): H1探测器记录的信号到达时间 (s)。
    t_l1 (float): L1探测器记录的信号到达时间 (s)。
    k_penalty (float): 惩罚因子，控制超出范围后得分下降的速度。

    返回:
    float: 到达时间一致性得分，范围在 (0, 1]。
    """
    # 光速 (km/s)
    C_LIGHT = 299792.458
    # LIGO Hanford (H1) 和 Livingston (L1) 探测器间距 (km)
    D_H1_L1 = 3002.15
    # 最大光行时间 (s)
    MAX_TRAVEL_TIME = D_H1_L1 / C_LIGHT

    delta_t = np.abs(t_h1 - t_l1)
    
    if delta_t <= MAX_TRAVEL_TIME:
        return 1.0
    else:
        # 计算超出部分并归一化
        excess_ratio = (delta_t - MAX_TRAVEL_TIME) / MAX_TRAVEL_TIME
        score = np.exp(-k_penalty * (excess_ratio ** 10))
        return score
    
def calculate_snr_score(rho_net: float, rho_H1, rho_L1) -> float:
    rho_threshold = np.sqrt(rho_H1**2 + rho_L1**2)
    score = 1.0 - np.exp(-(rho_net / rho_threshold) ** 2)
    return score

def calculate_parameter_score(params_h1: dict, params_l1: dict, alpha: float = 0.25) -> float:
    """
    计算参数一致性得分 (S_param)，不使用不确定度。

    此函数通过比较从两个探测器独立估计的相位和有效距离的相对差异来打分。

    参数:
    params_h1 (dict): 从H1数据估计的参数，应包含 {'phase': float, 'd_eff': float}。
    params_l1 (dict): 从L1数据估计的参数，应包含 {'phase': float, 'd_eff': float}。
    alpha (float): 有效距离的“可接受”相对差异尺度。

    返回:
    float: 参数一致性得分，范围在 [0, 1]。
    """
    # 1. 计算相位得分
    phi_h1 = params_h1['phase']
    phi_l1 = params_l1['phase']
    delta_phi = np.abs(phi_h1 - phi_l1)
    # 处理2pi周期性
    wrapped_delta_phi = min(delta_phi, 2 * np.pi - delta_phi)
    # S_phase = cos^2( d_phi / 2 )
    s_phase = np.cos(wrapped_delta_phi / 2) ** 2

    # 2. 计算有效距离得分
    d_h1 = params_h1['d_eff']
    d_l1 = params_l1['d_eff']
    # 避免分母为零
    if d_h1 + d_l1 == 0:
        # 如果距离都为0，认为它们是完全一致的
        relative_diff = 0.0
    else:
        # 计算归一化相对差异
        relative_diff = np.abs(d_h1 - d_l1) / ((d_h1 + d_l1) / 2)
    
    # S_dist = exp( -(rel_diff / alpha)^2 )
    s_dist = np.exp(-(relative_diff / alpha) ** 2)

    # 3. 组合得分
    s_param = (s_phase + s_dist) / 2
    return s_param

def detector_consistency_check(mf_result, cSNR, weight=(0.25, 0.4, 0.35), detail=False):
    timemasx_H1 = mf_result['timemasx_H1']
    phase_H1 = mf_result['phase_H1']
    d_eff_H1 = mf_result['d_eff_H1']
    SNRmax_H1 = mf_result['SNRmax_H1']
    timemasx_L1 = mf_result['timemasx_L1']
    phase_L1 = mf_result['phase_L1']
    d_eff_L1 = mf_result['d_eff_L1']
    SNRmax_L1 = mf_result['SNRmax_L1']
    params_H1 = {
        'phase': phase_H1, 
        'd_eff': d_eff_H1, 
    }
    params_L1 = {
        'phase': phase_L1, 
        'd_eff': d_eff_L1, 
    }

    S_t = calculate_time_score(timemasx_H1, timemasx_L1)
    S_c = calculate_snr_score(cSNR, SNRmax_H1, SNRmax_L1)
    S_p = calculate_parameter_score(params_H1, params_L1)
    S_tot = weight[0]*S_t + weight[1]*S_c + weight[2]*S_p
    if detail:
        return (S_tot, S_t, S_c, S_p)
    else:
        return S_tot
    
def run_analysis_pipeline(event_info, strain_not_L1):
    result = {}

    event_name = event_info['name']
    fs = event_info['fs']
    t_event = event_info['tevent']
    fband = event_info['fband']
    H1 = event_info['fn_H1']
    L1 = event_info['fn_L1']
    template_name = event_info['fn_template']

    template = load_template(template_name)
    datafreq = np.fft.fftfreq(template.size)*fs
    df = np.abs(datafreq[1] - datafreq[0])
    try:   dwindow = signal.windows.tukey(template.size, alpha=1./8)
    except: dwindow = signal.windows.blackman(template.size)
    template_fft = np.fft.fft(template*dwindow) / fs

    pre_progress = PreProcessing(event_name, fs, t_event, fband, H1, L1)
    data = pre_progress.main()

    strain_H1 = data['strain_H1']
    strain_L1 = strain_not_L1
    time = data['time_H1']
    dt = time[1] - time[0]
    f_H1 = data['f_H1']
    psd_H1 = data['psd_H1']
    f_L1, psd_L1 = get_PSD(strain_L1, fs, 4)
    selected_whiten_H1 = data['selected_whiten_H1']
    selected_whiten_L1 = band_filter(whiten(strain_not_L1, psd_L1, f_L1, dt), fband, fs)

    H1_fft = np.fft.fft(strain_H1*dwindow) / fs
    L1_fft = np.fft.fft(strain_L1*dwindow) / fs

    power_vec_H1 = np.interp(np.abs(datafreq), f_H1, psd_H1)
    power_vec_L1 = np.interp(np.abs(datafreq), f_L1, psd_L1)

    optimal_H1 = H1_fft * template_fft.conjugate() / power_vec_H1
    optimal_time_H1 = 2*np.fft.ifft(optimal_H1)*fs

    optimal_L1 = L1_fft * template_fft.conjugate() / power_vec_L1
    optimal_time_L1 = 2*np.fft.ifft(optimal_L1)*fs

    sigmasq_H1 = 1*(template_fft * template_fft.conjugate() / power_vec_H1).sum() * df
    sigma_H1 = np.sqrt(np.abs(sigmasq_H1))
    SNR_complex_H1 = optimal_time_H1/sigma_H1

    sigmasq_L1 = 1*(template_fft * template_fft.conjugate() / power_vec_L1).sum() * df
    sigma_L1 = np.sqrt(np.abs(sigmasq_L1))
    SNR_complex_L1 = optimal_time_L1/sigma_L1

    peaksample_H1 = int(strain_H1.size / 2)
    SNR_complex_H1 = np.roll(SNR_complex_H1,peaksample_H1)
    SNR_H1 = abs(SNR_complex_H1)

    peaksample_L1 = int(strain_L1.size / 2)
    SNR_complex_L1 = np.roll(SNR_complex_L1,peaksample_L1)
    SNR_L1 = abs(SNR_complex_L1)

    indmax_H1 = np.argmax(SNR_H1)
    timemax_H1 = time[indmax_H1]
    SNRmax_H1 = SNR_H1[indmax_H1]

    indmax_L1 = np.argmax(SNR_L1)
    timemax_L1 = time[indmax_L1]
    SNRmax_L1 = SNR_L1[indmax_L1]

    d_eff_H1 = sigma_H1 / SNRmax_H1
    horizon_H1 = sigma_H1/8
    phase_H1 = np.angle(SNR_complex_H1[indmax_H1])
    offset_H1 = (indmax_H1-peaksample_H1)

    d_eff_L1 = sigma_L1 / SNRmax_L1
    horizon_L1 = sigma_L1/8
    phase_L1 = np.angle(SNR_complex_L1[indmax_L1])
    offset_L1 = (indmax_L1-peaksample_L1)

    result.update(
        {
            'timemasx_H1': timemax_H1, 
            'SNRmax_H1': SNRmax_H1, 
            'phase_H1': phase_H1,
            'd_eff_H1': d_eff_H1, 
            'timemasx_L1': timemax_L1, 
            'SNRmax_L1': SNRmax_L1, 
            'phase_L1': phase_L1,
            'd_eff_L1': d_eff_L1, 
        }
    )

    strain = strain_H1 + strain_L1
    f, psd = get_PSD(strain, fs, 4)

    strain_fft = np.fft.fft(strain*dwindow) / fs

    power_vec = np.interp(np.abs(datafreq), f, psd)
    
    optimal = strain_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)*fs

    sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    peaksample = int(strain.size / 2)
    SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)

    indmax = np.argmax(SNR)
    timemax = time[indmax]
    SNRmax = SNR[indmax]

    score = detector_consistency_check(result, SNRmax)
    return score

def perform_time_slides(event_info, n_slides, delta_t_1=1, delta_t_2=3, extension=False, time_series=None):
    """
    执行N次随机时域平移，并记录下每次的背景得分。
    """
    event_name = event_info['name']
    fs = event_info['fs']
    t_event = event_info['tevent']
    fband = event_info['fband']
    H1 = event_info['fn_H1']
    L1 = event_info['fn_L1']
    if extension:
        strain_L1 = time_series
    else:
        pre_progress = PreProcessing(event_name, fs, t_event, fband, H1, L1)
        data = pre_progress.main()

        strain_L1 = data['strain_L1']

    background_scores = []
    print(f"正在执行 {n_slides} 次随机时域平移...")
    
    for i in range(n_slides):
        # 随机平移1到29秒，确保平移量足够大以打破真实关联
        random_shift_s = np.random.randint(delta_t_1, delta_t_2)
        random_shift_samples = random_shift_s * fs
        
        # 对L1噪声数据进行平移（环状）
        l1_shifted = np.roll(strain_L1, random_shift_samples)
        
        # 在平移后的纯噪声数据上运行分析流程
        score = run_analysis_pipeline(event_info, l1_shifted)
        background_scores.append(score)
        
        strain_L1 = l1_shifted
        
        if i % 10 == 0:
            print(f"  ...已完成 {i + 1}/{n_slides} 次平移")
            
    return background_scores

def circular_slicer(
    data: np.ndarray, 
    fs: float, 
    slice_duration: float, 
    start_time: float, 
    num_slices: int
) -> np.ndarray:
    """
    将数据视为一个环（头尾相连），从指定起点开始，连续切出指定数量的、
    等长的子序列。本实现为完全向量化的版本，效率极高。

    Args:
        data (np.ndarray): 一维原始时间序列数据，将被视为周期性的。
        fs (float): 采样频率 (Hz)。
        slice_duration (float): 每个子序列的目标时长 (秒)。
        start_time (float): 第一个切片的起始时刻 (秒)。
        num_slices (int): 需要切出的子序列总数。

    Returns:
        np.ndarray: 一个形状为 (num_slices, samples_per_slice) 的二维数组。
    """
    n_total_samples = data.shape[0]
    samples_per_slice = int(slice_duration * fs)

    if samples_per_slice > n_total_samples:
        print("错误: 单个切片的长度大于数据总长。")
        return np.array([])

    # 1. 计算每个切片在“无限长”的直线数据上的起始索引
    initial_start_index = int(start_time * fs)
    # 使用 np.arange 一次性生成所有切片的起始点
    slice_start_indices = initial_start_index + np.arange(num_slices) * samples_per_slice

    # 2. 生成一个二维的“相对索引”矩阵
    # 每一行都是 [0, 1, 2, ..., samples_per_slice-1]
    # shape: (num_slices, samples_per_slice)
    relative_indices_matrix = np.arange(samples_per_slice) + slice_start_indices[:, np.newaxis]

    # 3. 执行“环形”操作 (核心步骤)
    # 将直线索引通过取模运算，映射到环形的 [0, n_total_samples-1] 范围内
    circular_indices = relative_indices_matrix % n_total_samples

    # 4. 使用NumPy的高级索引，一步从原始数据中提取所有需要的值
    return data[circular_indices]