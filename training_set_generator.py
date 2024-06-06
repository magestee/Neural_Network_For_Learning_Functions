import csv

def generate_training_data():
    data = [(x, x**2) for x in range(1,6)]
    with open('training_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['input', 'output'])
        writer.writerow(data)

generate_training_data()
