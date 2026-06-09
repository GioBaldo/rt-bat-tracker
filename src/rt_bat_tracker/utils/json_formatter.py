import logging
import json


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            # "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "thread": record.threadName,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)
