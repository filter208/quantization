def analyze_timing(t_clk_ns, t_logic_total_ns, pipeline_stages, t_cq_ns=0.2, t_setup_ns=0.1, t_net_ns=0.5):
    """
    简易 FPGA 静态时序分析与流水线性能计算器
    """
    print(f"--- 硬件时序评估报告 ---")
    print(f"目标时钟周期: {t_clk_ns} ns ({1000/t_clk_ns:.1f} MHz)")
    print(f"总组合逻辑延迟: {t_logic_total_ns} ns")
    print(f"流水线级数: {pipeline_stages} 级")
    print("-" * 25)
    
    # 假设流水线能将逻辑延迟完美均分（实际中通常会有不平衡，这里作理论估算）
    t_logic_per_stage = t_logic_total_ns / pipeline_stages
    
    # 计算关键路径总延迟 (Data Path Delay)
    data_path_delay = t_cq_ns + t_logic_per_stage + t_net_ns
    
    # 计算所需的最小周期 (Minimum Period required)
    min_period = data_path_delay + t_setup_ns
    
    # 计算时序裕量 (Slack)
    slack = t_clk_ns - min_period
    
    # 计算理论最高频率 (Max Frequency)
    fmax_mhz = 1000 / min_period
    
    print(f"单级流水线逻辑延迟: {t_logic_per_stage:.2f} ns")
    print(f"关键路径总到达时间 (Data Arrival): {data_path_delay:.2f} ns")
    print(f"数据需求时间 (Data Required): {t_clk_ns - t_setup_ns:.2f} ns")
    print(f"建立时间裕量 (Setup Slack): {slack:.2f} ns")
    print(f"理论极限主频 (Fmax): {fmax_mhz:.1f} MHz")
    
    if slack >= 0:
        print("\n结论: \033[92m时序收敛 (MET)\033[0m. 你的设计满足时钟要求！")
    else:
        print("\n结论: \033[91m时序违例 (VIOLATED)\033[0m. 建议增加流水线级数或降低目标频率。")
        
# ==========================================
# 你可以在这里修改你的硬件参数进行模拟测试
# ==========================================
# 假设你想跑 250MHz (Tclk = 4.0ns)，近似逻辑本身需要 6ns
print("【方案 A：纯组合逻辑，无流水线】")
analyze_timing(t_clk_ns=4.0, t_logic_total_ns=6.0, pipeline_stages=1)

print("\n\n【方案 B：引入你重新设计的 2 级流水线】")
analyze_timing(t_clk_ns=4.0, t_logic_total_ns=6.0, pipeline_stages=2)