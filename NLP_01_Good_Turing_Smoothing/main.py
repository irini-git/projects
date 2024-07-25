from smoothing import Good_Turing_Smoothing

# Fishing scenario invented by Josh Goodman
Catch = {'carp':10,
         'perch':3,
         'whitefish':2,
         'trout':1,
         'salmon':1,
         'eel':1}

# Initiate class
probabilities = Good_Turing_Smoothing(Catch)

probabilities.calculate_c()

# Potential improvement : prompt for input
# Calculate probabilities
probabilities.calculate_probability('trout')
probabilities.calculate_probability('bass')

# ----------------- End