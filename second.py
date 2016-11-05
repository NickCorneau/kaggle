import csv as csv
import numpy as np

# Read in data
csv_file_object = csv.reader(open('csv/train.csv', 'rb')) 
header = csv_file_object.next()

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

# Set the fare ceiling which is the highest fare possible
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
data[data[0::,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

number_of_genders = 2
number_of_classes = len(np.unique(data[0::,2]))

# Initialize the survival table with all zeros
survival_table = np.zeros((number_of_genders, number_of_classes, number_of_price_brackets))

# Create survival table based on gender, class, and price bracket
for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):
        women_only_stats = data[(data[0::,4] == "female") \
                                & (data[0::,2].astype(np.float) == i+1) \
                                & (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                                & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size),1]
        men_only_stats = data[(data[0::,4] != "female") \
                              & (data[0::,2].astype(np.float) == i+1) \
                              & (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                              & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size),1]
        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

# handle nan's
survival_table[survival_table != survival_table] = 0

# If surival rate is over 50%, they survive. Else, they die.
survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

test_file = open('csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

predictions_file = open("csv/genderclassmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])


for row in test_file_object:
    for j in xrange(number_of_price_brackets):

        # convert fare to float
        try:
            row[8] = float(row[8])
        except:
            # if fare is NaN, use person's PClass to determine fare bin.
            bin_fare = 3 - (float(row[1])) # Source of error!

        # If person paid more than $40, bin them in highest price bracket
        if row[8] > fare_ceiling:
            bin_fare = number_of_price_brackets-1
            break
        
        # Otherwise bin them according to their respective fare.
        if ((row[8] >= j * fare_bracket_size) and \
           (row[8] < (j+1) * fare_bracket_size)):
            bin_fare = j
            break
        
        # Use the fare bin, Pclass, and gender to lookup their survival in the survival table.
        # Write out to predictions file.
        if row[3] == 'female':
            p.writerow([row[0], "%d" % int(survival_table[0, float(row[1])-1, bin_fare])])
        else:
            p.writerow([row[0], "%d" % int(survival_table[1, float(row[1])-1, bin_fare])])

test_file.close()
predictions_file.close()
print survival_table



# ~~ Sources of Errors ~~ #
# 1. Survival table says all men always die. Obviously not all men died.
# 2. If fare is N/A, we use the person's PClass to determine fare. Class doesn't necessarily determine fare.
