import logging
from logging import handlers


class Logger():
    def __init__(self,filename,level="debug",fmt_str="%(asctime)s - %(levelname)s - %(message)s",
                 date_str="%Y/%m/%d %H:%M:%S",when="D",backcount=3):
        self.setlevel = {"debug":logging.DEBUG,
                         "info":logging.INFO,
                         "warning":logging.WARNING,
                         "error":logging.ERROR}
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.setlevel[level])
        format_str = logging.Formatter(fmt=fmt_str,datefmt=date_str)

        sh = logging.StreamHandler()
        sh.setFormatter(format_str)

        th = handlers.TimedRotatingFileHandler(filename,when=when,backupCount=backcount,encoding="utf-8")
        th.setFormatter(format_str)

        self.logger.addHandler(sh)
        self.logger.addHandler(th)

if __name__ == '__main__':
    log = Logger("./test.log")
    log.logger.info("hello")
    log.logger.info("你好")
