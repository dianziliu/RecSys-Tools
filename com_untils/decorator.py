import time
import inspect
from functools import wraps
from typing import Tuple

step=4
chap=" "*step
deep=0


def arg_value(arg_name, f, args, kwargs):
    if arg_name in kwargs:
        return kwargs[arg_name]

    i = f.__code__.co_varnames.index(arg_name)
    if i < len(args):
        return args[i]

    return inspect.signature(f).parameters[arg_name].default


def funcion_logger(begin_message: str = None, log_args: Tuple[str] = None,
           end_message: str = None, log_time: bool = True):
    """
        一个函数包装器，主要功能有：
        1. 打印运行函数名
        2. 在开始和结束时打印信息
        3. 显示指定参数的值
    """
    def logger_decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            global deep
            
            if begin_message is not None:
                print(chap*deep+begin_message, end='\n' if log_args is None else '  ')
            deep+=1
            if log_args is not None:
                arg_logs = [arg_name + '=' + str(arg_value(arg_name, f, args, kwargs)) for arg_name in log_args]
                print(', '.join(arg_logs))

            start_time = time.time()
            result = f(*args, **kwargs)
            spent_time = time.time() - start_time
            
            deep-=1
            if end_message is not None:
                
                print(chap*deep+end_message)
                
            if log_time:
                print('（耗时', spent_time, '秒）', sep='')

            return result
        return decorated
    return logger_decorator

@funcion_logger("test",("x"),"finishede")
def test(x):
    print(x)
    test2("1")



@funcion_logger("test",("x"),"finishede")
def test2(x):
    print("helo",x)
if __name__=='__main__':

    test("hello,world")
    pass
