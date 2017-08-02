linear_model = polyfit(sqft_living(2:end),price(2:end),1)
(800000 - linear_model(2))/linear_model(1)