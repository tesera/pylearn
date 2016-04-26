'''
This module sets up logging to cloudwatch.
Log levels: debug, info, warning, error, critical.
'''
import logging
import time
import boto3
import botocore

class CloudWatchHandler(logging.Handler):

    def __init__(self, log_group_name, log_stream_name, aws_region='us-east-1'):
        logging.Handler.__init__(self)

        self.cw_client = boto3.client('logs', aws_region);
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name

        try:
            self.cw_client.create_log_group(logGroupName=log_group_name)
        except botocore.exceptions.ClientError as error:
            pass

        try:
            self.cw_client.create_log_stream(logGroupName=log_group_name, logStreamName=self.log_stream_name)
        except botocore.exceptions.ClientError as error:
            pass

    def emit(self, record):
        if (hasattr(self, 'response')):
            self.response = self.cw_client.put_log_events(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name,
                logEvents=[
                    {
                        'timestamp': int(time.time())*1000,
                        'message': self.format(record)
                    },
                ],
                sequenceToken=self.response['nextSequenceToken']
            )
        else:
            self.response = self.cw_client.put_log_events(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name,
                logEvents=[
                    {
                        'timestamp': int(time.time())*1000,
                        'message': self.format(record)
                    },
                ]
            )
