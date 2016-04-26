import logging
import logging.handlers
from cloudwatch import CloudWatchHandler
import uuid
import os

log_level = os.getenv('LOGGING_LEVEL', 'INFO')
aws_logs = os.getenv('AWS_LOGS', 'false')
aws_logs_group = os.getenv('AWS_LOGS_GROUP', 'pylearn')
aws_logs_stream_prefix = os.getenv('AWS_LOGS_STREAM_PREFIX', 'py')
aws_logs_stream = "%s-%s" % (aws_logs_stream_prefix, str(uuid.uuid4()))


logger = logging.getLogger('pylearn')

if aws_logs.lower() == 'true':
    logging.handlers.CloudWatchHandler = CloudWatchHandler
    cloudWatchHandler = logging.handlers.CloudWatchHandler(aws_logs_group, aws_logs_stream)
    cloudWatchHandler.setLevel(log_level)
    logger.addHandler(cloudWatchHandler)

logging.basicConfig(filename='pylearn.log', level=log_level)
