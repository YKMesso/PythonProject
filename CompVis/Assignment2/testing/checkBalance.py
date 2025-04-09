import os

#Check if balanced
def count_images(dir):
    class_counts = {}
    for class_name in os.listdir(dir):
        class_path = os.path.join(dir, class_name)
        if os.path.isdir(class_path):
            count = len([file for file in os.listdir(class_path) if file.lower().endswith('.jpeg')])
            class_counts[class_name] = count
    return class_counts

train_dir = '/CompVis/Assignment2/chest_xray/train'
test_dir = '/CompVis/Assignment2/chest_xray/test'


train_count = count_images(train_dir)
test_count = count_images(test_dir)

print("Training Set Distribution:", train_count)
print("Test Set Distribution:", test_count)
