import os
from multiprocessing import Process
import argparse

HOSTs = [
    'g4001',
    'g4002',
    'g4003',
    'g4004',
    'g4005',
    'g4006',
    'g4007',
    'g4008',
    'g3021',
    'g3022',
    'g3023',
    'g3024',
    'g3025',
    'g3026',
    'g3027',
    'g3028',
    'g3029',
    'g1004',
    'g1006',
]
LOG_DIR=f'{os.getcwd()}/results'

class Cluster_Profiler(Process):
    def __init__(self, host, cmd):
        super(Cluster_Profiler, self).__init__()
        self.host = host
        self.cmd = cmd
    
    def run(self):
        os.system(f'ssh {self.host} "{self.cmd} > {LOG_DIR}/{self.host}.log"')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default="nvidia-smi")

    args = parser.parse_args()
    return args

def main(args):
    print(f'cmd: {args.cmd}')
    ps = []
    for HOST in HOSTs:
        p = Cluster_Profiler(HOST, args.cmd) #实例化进程对象
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    
    # combine all logs
    OUTPUT_FILE = f'{LOG_DIR}/cluster_profiler_{"".join(args.cmd.split())}.log'
    print(f'output file: {OUTPUT_FILE}')
    os.system(f'rm -f {OUTPUT_FILE}')
    os.system(f'touch {OUTPUT_FILE}')
    for HOST in HOSTs:
        os.system(f'echo "================== {HOST} ==================" >> {OUTPUT_FILE}')
        os.system(f'cat {LOG_DIR}/{HOST}.log >> {OUTPUT_FILE}')
        os.system(f'echo "\n\n" >> {OUTPUT_FILE}')
        os.system(f'rm -f {LOG_DIR}/{HOST}.log')
    
if __name__ == '__main__':
    main(parse_args())