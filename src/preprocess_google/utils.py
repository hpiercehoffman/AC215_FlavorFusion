import json
import numpy as np

RESTS = np.array(['American restaurant', 'Angler fish restaurant',
       'Armenian restaurant', 'Asian fusion restaurant',
       'Asian restaurant', 'Australian restaurant', 'Austrian restaurant',
       'Barbecue restaurant', 'Breakfast restaurant', 'Brunch restaurant',
       'Buffet restaurant', 'Burrito restaurant',
       'Cheesesteak restaurant', 'Chicken restaurant',
       'Chicken wings restaurant', 'Chinese noodle restaurant',
       'Chinese restaurant', 'Chophouse restaurant',
       'Continental restaurant', 'Delivery Chinese restaurant',
       'Delivery Restaurant', 'Dessert restaurant',
       'Down home cooking restaurant', 'European restaurant',
       'Family restaurant', 'Fast food restaurant', 'Filipino restaurant',
       'Fine dining restaurant', 'Fish & chips restaurant',
       'German restaurant', 'Gluten-free restaurant', 'Greek restaurant',
       'Hamburger restaurant', 'Hawaiian restaurant',
       'Health food restaurant', 'Hoagie restaurant',
       'Hot dog restaurant', 'Indian restaurant', 'Irish restaurant',
       'Israeli restaurant', 'Italian restaurant', 'Japanese restaurant',
       'Korean restaurant', 'Latin American restaurant',
       'Lebanese restaurant', 'Lunch restaurant', 'Meat dish restaurant',
       'Mediterranean restaurant', 'Mexican restaurant',
       'Mexican torta restaurant', 'Middle Eastern restaurant',
       'Mongolian barbecue restaurant', 'New American restaurant',
       'Organic restaurant', 'Pan-Asian restaurant',
       'Peruvian restaurant', 'Pho restaurant', 'Pizza restaurant',
       'Ramen restaurant', 'Restaurant', 'Restaurant or cafe',
       'Restaurant supply store', 'Rice restaurant', 'Seafood restaurant',
       'Small plates restaurant', 'Soul food restaurant',
       'Soup restaurant', 'Southeast Asian restaurant',
       'Southern restaurant (US)', 'Southwestern restaurant (US)',
       'Spanish restaurant', 'Sushi restaurant', 'Taco restaurant',
       'Taiwanese restaurant', 'Takeout Restaurant', 'Takeout restaurant',
       'Tex-Mex restaurant', 'Thai restaurant',
       'Traditional American restaurant', 'Traditional restaurant',
       'Vegan restaurant', 'Vegetarian restaurant',
       'Venezuelan restaurant', 'Vietnamese restaurant',
       'Western restaurant'], dtype='<U31')

def parse(path):
    g = open(path, 'r')
    for l in g:
        yield json.loads(l)