"""
训练监控脚本：每2分钟汇报进度，智能检测卡住/死循环
"""
import subprocess
import time
import os
import sys


def get_cpu_time(pid):
    """通过 tasklist 获取进程 CPU 时间（Windows）"""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/V"],
            capture_output=True, text=True, encoding="gbk", errors="ignore", timeout=10
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            # CSV 格式，查找 CPU Time 字段
            header = [h.strip('"') for h in lines[0].split(",")]
            data = [d.strip('"') for d in lines[1].split(",")]
            for i, h in enumerate(header):
                if "CPU Time" in h and i < len(data):
                    return data[i]
    except Exception:
        pass
    return None


def monitor():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_log = os.path.join(base_dir, "train_output.log")
    err_log = os.path.join(base_dir, "train_error.log")
    progress_log = os.path.join(base_dir, "progress.log")

    for f in [train_log, err_log, progress_log, "test_output.log", "test_error.log"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"[{time.strftime('%H:%M:%S')}] 启动训练...")
    proc = subprocess.Popen(
        [sys.executable, "-u", "src/train.py"],
        stdout=open(train_log, "w", encoding="utf-8"),
        stderr=open(err_log, "w", encoding="utf-8"),
        cwd=base_dir,
        env=env
    )

    last_size = 0
    last_err_size = 0
    last_cpu_time = None
    stall_count = 0
    max_stall = 15  # 30分钟无 stdout 更新视为卡住
    check_interval = 120

    with open(progress_log, "w", encoding="utf-8") as plog:
        plog.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 监控启动，训练 PID: {proc.pid}\n")
        plog.flush()

        while proc.poll() is None:
            time.sleep(check_interval)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            size = os.path.getsize(train_log) if os.path.exists(train_log) else 0
            err_size = os.path.getsize(err_log) if os.path.exists(err_log) else 0
            cpu_time = get_cpu_time(proc.pid)

            # 读取最近输出
            try:
                with open(train_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    last_lines = "".join(lines[-25:]) if lines else "(无输出)"
            except Exception as e:
                last_lines = f"读取日志失败: {e}"

            report = (
                f"\n{'='*60}\n"
                f"[{timestamp}] 训练进度汇报\n"
                f"PID: {proc.pid} | 日志: {size} bytes | 错误日志: {err_size} bytes\n"
                f"CPU Time: {cpu_time or 'N/A'}\n"
                f"最近输出:\n{last_lines}\n"
                f"{'='*60}\n"
            )
            plog.write(report)
            plog.flush()
            print(report, end="")

            # 判断是否有进展
            has_progress = False
            if size > last_size:
                has_progress = True
            if err_size > last_err_size:
                has_progress = True
            if cpu_time and last_cpu_time and cpu_time != last_cpu_time:
                has_progress = True

            if has_progress:
                if stall_count > 0:
                    plog.write(f"[{timestamp}] 训练恢复进展，重置 stall 计数\n")
                    plog.flush()
                    print(f"[{timestamp}] 训练恢复进展，重置 stall 计数")
                stall_count = 0
                last_size = size
                last_err_size = err_size
            else:
                stall_count += 1
                warn_msg = f"[{timestamp}] 警告: 已连续 {stall_count}/{max_stall} 次无输出 (stdout={size}, stderr={err_size}, cpu={cpu_time})\n"
                plog.write(warn_msg)
                plog.flush()
                print(warn_msg, end="")
                if stall_count >= max_stall:
                    kill_msg = f"[{timestamp}] 训练长时间无响应，强制终止...\n"
                    plog.write(kill_msg)
                    plog.flush()
                    print(kill_msg, end="")
                    proc.terminate()
                    time.sleep(5)
                    if proc.poll() is None:
                        proc.kill()
                    break

            last_cpu_time = cpu_time

        exit_code = proc.poll()
        end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        finish_msg = f"[{end_ts}] 训练结束，退出码: {exit_code}\n"
        plog.write(finish_msg)
        plog.flush()
        print(finish_msg, end="")

        if exit_code == 0:
            run_msg = f"[{end_ts}] 训练成功，开始运行预测...\n"
            plog.write(run_msg)
            plog.flush()
            print(run_msg, end="")

            test_proc = subprocess.Popen(
                [sys.executable, "-u", "src/test.py"],
                stdout=open(os.path.join(base_dir, "test_output.log"), "w", encoding="utf-8"),
                stderr=open(os.path.join(base_dir, "test_error.log"), "w", encoding="utf-8"),
                cwd=base_dir,
                env=env
            )
            test_proc.wait()
            end_ts2 = time.strftime("%Y-%m-%d %H:%M:%S")
            test_msg = f"[{end_ts2}] 预测完成，退出码: {test_proc.returncode}\n"
            plog.write(test_msg)
            plog.flush()
            print(test_msg, end="")
        else:
            err_msg = f"[{end_ts}] 训练异常退出 (exit_code={exit_code})，跳过预测\n"
            plog.write(err_msg)
            plog.flush()
            print(err_msg, end="")


if __name__ == "__main__":
    monitor()
