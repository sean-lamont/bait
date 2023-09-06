"""Open source mocks for reading/writing recordIO."""


from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
from google.protobuf import text_format
from typing import Text

# def read_protos_from_recordio(pattern: Text, proto_class):
#   del pattern
#   del proto_class
#   assert False, 'Recordio input is not supported.'

# def write_protos_to_recordio(filename: Text, protos):
#   del filename
#   del protos
#   assert False, 'Recordio output is not supported.'



# implementation with standard open context manager, and protobuf text_format
def read_protos_from_recordio(pattern: Text, proto_class):
  proto_class = proto_class()

  with open(pattern) as f:
    text_format.MergeLines(f, proto_class)
  return proto_class

def write_protos_to_recordio(filename: Text, protos):
  with open(filename, 'w') as f:
    f.write(text_format.MessageToString(protos))
