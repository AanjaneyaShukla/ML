import random
import sys

def gibbs_rain(n):
    random.uniform(0, 1)

    #Constraints
    sprinkler = True
    wet = True

    # Initialise the samples for cloudy and rain as true
    cloudy = True
    rain = True

    #Number of samples generated
    num_samples = 0

    #Number of samples with cloudy true
    num_cloudy = 0
    num_rain = 0

    #Probabilities for drawing samples
    p_cloudy_given_rain = 0.4444
    p_rain_given_cloudy = 0.814815
    p_cloudy_given_no_rain = 0.04762
    p_rain_given_no_cloudy = 0.215686

    flag = True
    #Number of samples with rain true
    for i in range(n):
        num_samples = num_samples + 1
        # if true this should sample cloudy else rain
        if (flag):
            if (rain):
                if (random.random() <= p_cloudy_given_rain):
                    cloudy = True
                    num_cloudy = num_cloudy + 1
                else:
                    cloudy = False
            else:
                if (random.random() <= p_cloudy_given_no_rain):
                    cloudy = True
                    num_rain = num_rain + 1
                else:
                    cloudy = False

        else:
            if (cloudy):
                if (random.random() <= p_rain_given_cloudy):
                    rain = True
                    num_rain = num_rain + 1
                else:
                    rain = False
            else:
                if (random.random() <= p_rain_given_no_cloudy):
                    num_rain = num_rain + 1
                    rain = True
                else:
                    rain = False

        flag = not flag

    print num_samples
    print (num_rain/(num_samples*1.0))
gibbs_rain(int(sys.argv[1]))