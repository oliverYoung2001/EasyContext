import schedule
import os
from datetime import datetime


def profiler_func(FILE_NAME):
    os.system(f'echo {datetime.now()} >> {FILE_NAME}')
    os.system(f'sinfo >> {FILE_NAME}')
    
def main():
    FILE_NAME = './results/sinfo.log'
    os.system(f'rm -f {FILE_NAME}')
    schedule.every(10).minutes.do(profiler_func, FILE_NAME)
    # schedule.every(2).seconds.do(profiler_func, FILE_NAME)
    while True:
        schedule.run_pending()
    
    
if __name__ == '__main__':
    main()