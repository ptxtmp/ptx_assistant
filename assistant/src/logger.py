import logging
import sys

FORMAT = "[%(levelname)s|%(filename)s:%(lineno)d] - %(threadName)s %(asctime)s ->   %(message)s"
formatter = logging.Formatter(FORMAT)

log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.DEBUG)


class UnbufferedStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)

    def emit(self, record):
        super().emit(record)
        self.flush()


h1 = UnbufferedStreamHandler(stream=sys.stdout)
h1.setLevel(logging.DEBUG)
# Filter out everything that is above INFO level (WARN, ERROR, ...)
h1.addFilter(lambda record: record.levelno <= logging.INFO)
h1.setFormatter(formatter)
log.addHandler(h1)

h2 = UnbufferedStreamHandler(stream=sys.stderr)
# Take only warnings and error logs
h2.setLevel(logging.WARNING)
h2.setFormatter(formatter)
log.addHandler(h2)

log.info(f"App logger initialized with log level `{logging.getLevelName(log.getEffectiveLevel())}`")
