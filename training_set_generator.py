import csv

def generate_training_data(rows):
    data = [(x, x**2) for x in range(1,rows)]
    with open('training_set.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['input', 'output'])
        writer.writerows(data)

generate_training_data(100)
