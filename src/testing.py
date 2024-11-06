import subprocess
import ast

test_images = subprocess.run(['ls', 'test/images'], capture_output=True, text=True)
test_images = test_images.stdout.splitlines()

labels = subprocess.run(['ls', 'test/labels'], capture_output=True, text=True)
labels = labels.stdout.splitlines()

labels_content = []
for i in labels:
    content = subprocess.run(['cat', 'test/labels/' + i], capture_output=True, text=True)
    lines = content.stdout.strip().splitlines()
    
    first_column = [int(line.split()[0]) for line in lines]
    first_column.sort()
    labels_content.append(first_column)

amount = 44
for index, image in enumerate(test_images):
    output = subprocess.run(
        ['python3', 'src/predict_param.py', image],
        capture_output=True,
        text=True
    )
    output = output.stdout.strip()
    try:
        result_list = ast.literal_eval(output[output.find('['):output.find(']')+1])
        result_list.sort()
        
    except (ValueError, SyntaxError):
        print("Error: Could not parse output as a list")

    print(f"{result_list} : {labels_content[index]}")
    if result_list != labels_content[index]:
        amount =  amount - 1

accuracy = (100 * amount) / 44
print(accuracy)








