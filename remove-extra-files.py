# THIS REMOVES IMAGES FROM THE HANDS FILE SO THAT WE HAVE LESS STUFF
# DO NOT RUN THIS!!!! OR ELSE YOU GOTTA REDOWNLOAD SHIT

from pathlib import Path
import random

def delete_images(directory, number_of_images, extension='jpg'):
    images = Path(directory).glob(f'*.{extension}')
    for image in random.sample(list(images), number_of_images):
        image.unlink()

total_num_files = 11076 # total number of files
keep = 200 # want to keep 200


'''
Only uncomment if you're trying to delete shit
'''
# delete_images('real_hands', total_num_files - keep)