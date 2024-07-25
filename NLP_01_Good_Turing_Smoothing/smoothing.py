import pandas as pd


class Good_Turing_Smoothing:
    def __init__(self, catch):
        self.catch = catch
        self.total = sum(self.catch.values())
        self.intro_message()

    def intro_message(self):
        """Print catch summary on the screen"""
        print(f'Catch of the day\n{self.catch}\n')

    def calculate_c(self):
        """Calculate c for all existing values.
        Larger corpus might be more interesting to use.
        """
        # Unique counts for catch
        count = sorted(list(set(self.catch.values())))

        # Good-Turing c = (c+1) N_c+1 / Nc
        c_asteriks = []

        for c in count:
            c_plus_one = c + 1  # count of catch plus one
            Nc_plus_one = len([k for k, v in self.catch.items() if
                               v == c_plus_one])  # frequency of c + 1 : how many spices were caught c+1 times
            Nc = len([k for k, v in self.catch.items() if v == c])
            discount_factor = c_plus_one * Nc_plus_one / Nc

            c_asteriks.append(discount_factor)

        df = pd.DataFrame({'count': count, 'c_asteriks': c_asteriks})

        print(f'Resulting Good-Turing numbers\n{df}\n')

    def calculate_probability(self, x):
        """
        Return MLE and Good-Turing probability for next fish
        :param x: next fish
        :return: print message with two probabilities
        """
        P_mle = self.calculate_mle_probability(x)
        P_GT = self.calculate_good_turing_probability(x)
        print(f'How likely is that next spices is {x}'
              f'\n   Maximum Likelihood Estimate: {round(P_mle, 2)}'
              f'\n   Good-Turing: {round(P_GT, 2)}\n')

    def calculate_mle_probability(self, x):
        """
        Calculate Maximum Likelihood Estimate probability
        How likely is that next species is ... trout / new
        :param x: next species
        :return: classic (non-Good-Turing) probability
        """
        if x in self.catch:
            # Not new species
            result = self.catch[x] / self.total
        else:
            # New species
            result = 0

        return result

    def calculate_good_turing_probability(self, x):
        """
        Calculate Good-Turing probability
        How likely is that next species is ... trout / new
        Use estimate of things we saw once to estimate new things
        :param x: next species
        :return: Good-Turing probability
        """
        if x in self.catch:
            # Not new species
            # Total probability mass must be reduced
            # Potential improvement: use resulting Good-Turing numbers
            c = self.catch[x]  # count of catch
            c_plus_one = self.catch[x] + 1  # count of catch plus one
            Nc_plus_one = len([k for k, v in self.catch.items() if
                               v == c_plus_one])  # frequency of c + 1: how many spices were caught c+1 times
            Nc = len([k for k, v in self.catch.items() if v == c])
            discount_factor = c_plus_one * Nc_plus_one / Nc

            result = discount_factor / self.total

        else:
            # New species
            # Frequency of frequency one - occurred once
            N1 = len([k for k, v in self.catch.items() if v == 1])

            result = N1 / self.total

        return result

# ----------------- End
