from root (deepmath-light) dir:

   ` python -m grpc_tools.protoc -I=. --python_out=. ./deepmath/deephol/deephol.proto`
    `python -m grpc_tools.protoc -I=. --python_out=. ./deepmath/proof_assistant/proof_assistant.proto --grpc_python_out=.`
