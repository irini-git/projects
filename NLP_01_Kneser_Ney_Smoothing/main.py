from kn_smoothing import Kneser_Ney_Smoothing

# Initiate class
smoothing = Kneser_Ney_Smoothing()

# Optional: plot wordcloud
smoothing.plot_wordcloud()

# Calculate probability of the text
probability = round(smoothing.calculate_probability(),2)
print(f'{probability}')

# End -------------------------------

