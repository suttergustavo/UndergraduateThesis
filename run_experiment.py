import argparse
import os
import yaml

import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.image import ImageDataGenerator

from datautils import (examples_per_class, prepare_path_list,
                       prepare_path_list_celeba)
from metrics import FID, DendrogramDistance, InceptionScore


def get_data_loaders(options):
    real_data = options['real_data']
    real_loader = ImageDataGenerator(rescale=1/255)

    if options['dataset'] == 'CelebA': 
        real_file_list = prepare_path_list_celeba(options['dataset_path'],
                                                  options['attr_list_path'],
                                                  real_data['classes_dist'],
                                                  repeat=real_data['repeat'],
                                                  seed=options['random_seed'])
    else:
        real_file_list = prepare_path_list(options['dataset_path'],
                                        real_data['classes_dist'],
                                        repeat=real_data['repeat'],
                                        seed=options['random_seed'])
    


    real_it = real_loader.flow_from_dataframe(real_file_list,
                                            target_size=options['target_size'],
                                            batch_size=options['batch_size'],
                                            class_mode='categorical')

    real_it_pixel = real_loader.flow_from_dataframe(real_file_list,
                                                    target_size=options['target_size_pixel_space'],
                                                    batch_size=options['batch_size'],
                                                    class_mode='categorical')

                
    fake_its = {}
    fake_its_pixel = {}
    for name, fake_it_specs in options['fake_data'].items():
        if options['dataset'] == 'CelebA':
            file_list = prepare_path_list_celeba(options['dataset_path'],
                                                 options['attr_list_path'],
                                                 fake_it_specs['classes_dist'],
                                                 repeat=fake_it_specs['repeat'],
                                                 seed=options['random_seed'])
        else:
            file_list = prepare_path_list(options['dataset_path'],
                                          fake_it_specs['classes_dist'],
                                          repeat=fake_it_specs['repeat'],
                                          seed=options['random_seed'])

        fake_loader = ImageDataGenerator(rescale=1/255)
        fake_it = fake_loader.flow_from_dataframe(file_list,
                                                  target_size=options['target_size'],
                                                  batch_size=options['batch_size'],
                                                  class_mode='categorical')
        
        fake_it_pixel = fake_loader.flow_from_dataframe(file_list,
                                                        target_size=options['target_size_pixel_space'],
                                                        batch_size=options['batch_size'],
                                                        class_mode='categorical')
    

        fake_its[name] = fake_it
        fake_its_pixel[name] = fake_it_pixel

    return real_it, real_it_pixel, fake_its, fake_its_pixel


def save_results(output_path, options, results_df, real_distances, fake_distances, verbose):
    if verbose:
        print(f'Saving results in {output_path}')
    
    os.makedirs(output_path, exist_ok=True)
    
    results_df.to_csv(os.path.join(output_path, 'results.csv'), index=False)
    
    with open(os.path.join(output_path, 'real_distances.pickle'), 'wb') as f:
        pickle.dump(real_distances, f)

    with open(os.path.join(output_path, 'fake_distances.pickle'), 'wb') as f:
        pickle.dump(fake_distances, f)

    with open(os.path.join(output_path, 'setup.yaml'), 'w') as f:
        yaml.safe_dump(options, f)


if __name__ == '__main__':
    # Setting up the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('setup', type=str, help='Yaml specification of the experiment to be performed')
    parser.add_argument('output', type=str, help='Directory where the output folder must be saved')
    args = parser.parse_args()

    # Reading the setup fileoutput
    with open(args.setup) as f:
        options = yaml.safe_load(f)

    print(options)
    print('\n')

    # Getting real and fake data loaders
    real_it, real_it_pixel, fake_its, fake_its_pixel = get_data_loaders(options)

    # Instantiating the objects that calculate our metrics
    fid = FID(lat_as_input=True, verbose=options['verbose'])
    is_metric = InceptionScore(lat_as_input=True, verbose=options['verbose'])
    dendro_last = DendrogramDistance(use_layer='last', agg_type='both', verbose=options['verbose'])
    dendro_hidden = DendrogramDistance(use_layer='hidden', agg_type='both', verbose=options['verbose'])
    dendro_pixel = DendrogramDistance(use_layer='pixel', agg_type='both', verbose=options['verbose'])

    # Fitting the metrics (that need to be fitted) to the real data
    real_hidden = dendro_hidden.fit(real_it, return_lat=True)
    fid.fit(real_hidden)
    dendro_last.fit(real_it)
    dendro_pixel.fit(real_it_pixel)

    # Saving dendrogram distances of the real data
    real_distances = {}
    real_distances['hidden'] = dendro_hidden.dendro_real.distances_
    real_distances['last'] = dendro_last.dendro_real.distances_
    real_distances['pixel'] = dendro_last.dendro_real.distances_

    # Where the results will be stored
    results = []
    fake_distances = {}

    # Evaluating with "fake" data  
    for i, (name, fake_it) in enumerate(fake_its.items()):
        if options['verbose']:
            print(f'Processing {name} ({i+1}/{len(fake_its)})')

        # Doing the actual evaluation
        dendro_hidden_score, fake_hidden = dendro_hidden.evaluate(fake_it, return_lat=True)
        dendro_last_score, fake_last = dendro_last.evaluate(fake_it, return_lat=True)
        dendro_pixel_score = dendro_pixel.evaluate(fake_its_pixel[name])
        is_score = is_metric.evaluate(fake_last)
        fid_score = fid.evaluate(fake_hidden)
        
        # Saving results
        results.append([
            name,
            dendro_hidden_score['mean'],
            dendro_hidden_score['max'],
            dendro_last_score['mean'],
            dendro_last_score['max'],
            dendro_pixel_score['mean'],
            dendro_pixel_score['max'],
            is_score,
            fid_score
        ])

        # Saving dendrogram distances of the current fake data
        fake_distances[name] = {
            'hidden': dendro_hidden.dendro_fake.distances_,
            'last': dendro_last.dendro_fake.distances_,
            'pixel': dendro_pixel.dendro_fake.distances_
        }

    # Creating a dataframe to store the results
    columns=['Data Partition',
            'Dendrogram Hidden (Mean)',
            'Dendrogram Hidden (Max)',
            'Dendrogram Last (Mean)',
            'Dendrogram Last (Max)',
            'Dendrogram Pixel (Mean)',
            'Dendrogram Pixel (Max)',
            'Inception Score',
            'FID']

    results_df = pd.DataFrame(results, columns=columns)

    # Saving all the results to be analyzed later
    save_results(args.output,
                 options, 
                 results_df, 
                 real_distances, 
                 fake_distances,
                 options['verbose'])

