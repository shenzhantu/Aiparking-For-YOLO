import onnx
model = onnx.load(r'D:\Aiparking\Aiparking For YOLO\models\best.onnx')
print('=== Inputs ===')
for inp in model.graph.input:
    print(f'  Name: {inp.name}')
    shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f'  Shape: {shape}')
print()
print('=== Outputs ===')
for out in model.graph.output:
    print(f'  Name: {out.name}')
    shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f'  Shape: {shape}')
print()
print('Opset:', model.opset_import[0].version)
