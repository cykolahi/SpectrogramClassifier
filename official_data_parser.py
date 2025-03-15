import pandas as pd
import geopy
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import utils.utils as utils

class AudioLocationParser:
    def __init__(self):
        """Initialize the parser with Nominatim geocoder"""
        self.geolocator = Nominatim(user_agent="my_geopy_app")
        self.audio_paths = []
        self.coordinates = []
        self.countries = []
        self.artist_location_dict = {}


    def get_coordinates_with_audio_paths(self, audio_path):
        tracks = utils.load('/projects/dsci410_510/Kolahi_data_temp/fma_metadata/tracks.csv')
        medium = tracks[tracks['set', 'subset'] <= 'medium']
        #small = small[small['artist_latitude'].notna()]
        self.artist_location_dict = {}
        for track_id, artist in medium['artist'].iterrows():
            lat = artist['latitude']
            lon = artist['longitude']
            if pd.notna(lat) and pd.notna(lon):
                self.artist_location_dict[track_id] = (lat, lon)

        self.audio_paths = []
        for track_id, coordinates in self.artist_location_dict.items():
            #oordinates = coordinates.split(',')
            #latitude = coordinates[0]
            #longitude = coordinates[1]
            audio_path = utils.get_audio_path('/projects/dsci410_510/Kolahi_data_temp/fma_medium', track_id)
            self.audio_paths.append(audio_path)
        return self.artist_location_dict, self.audio_paths



    def get_country_from_coordinates(self, latitude, longitude, max_retries=3):
        """
        Get country name from latitude/longitude coordinates
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            max_retries (int): Number of retries for failed requests
            
        Returns:
            str: Country name or error status
        """
        for attempt in range(max_retries):
            try:
                location = self.geolocator.reverse((latitude, longitude))
                if location is None:
                    return "Unknown"
                
                address = location.raw['address']
                country = address.get('country', 'Unknown')
                return country
                
            except GeocoderTimedOut:
                if attempt == max_retries - 1:  # Last attempt
                    return "Timeout"
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                print(f"Error for coordinates {latitude}, {longitude}: {str(e)}")
                return "Error"

    def parse_locations(self, coordinates_df, save_path=None, save_interval=10):
        """
        Parse coordinates file and get corresponding countries
        
        Args:
            coordinates_df (pd.DataFrame): DataFrame with coordinates
            save_path (str): Path to save output CSV
            save_interval (int): How often to save progress
            
        Returns:
            pd.DataFrame: DataFrame with audio paths and countries
        """
        countries = []
        processed_data = []
        
        for i, (track_id, (latitude, longitude)) in enumerate(coordinates_df.items()):
            print(f"Processing {i+1}/{len(coordinates_df)}: {latitude}, {longitude}")
            
            country = self.get_country_from_coordinates(latitude, longitude)
            audio_path = self.audio_paths[i]
            processed_data.append({'audio_path': audio_path, 'country': country})
            
            # Add delay between requests
            time.sleep(1)
            
            if save_path and (i + 1) % save_interval == 0:
                temp_df = pd.DataFrame(processed_data, columns=['track_id', 'audio_path', 'country'])
                temp_df.to_csv(save_path.replace('.csv', '_partial.csv'), index=False)

            # Save progress periodically
        
        # Create final dataframe
        df = pd.DataFrame(processed_data, columns=['track_id', 'audio_path', 'country'])

        
        # Save final results if path provided
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nCompleted! Results saved to {save_path}")
            print(f"Successfully processed {len([c for c in countries if c != 'Timeout' and c != 'Error'])} locations")

        return df
    

    def filter_and_balance_dataset(self, df, min_samples=30, target_samples=None, balance_classes=False):
        """
        Filter countries with minimum samples and optionally balance classes
        
        Args:
            df (pd.DataFrame): DataFrame with audio_path and country columns
            min_samples (int): Minimum samples required per country
            target_samples (int): Target number of samples per country for balancing
            
        Returns:
            pd.DataFrame: Filtered and balanced DataFrame
        """
        # Remove error/unknown countries
        df = df[~df['country'].isin(['Error', 'Unknown', 'Timeout'])]
        
        # Filter countries with minimum samples
        country_counts = df['country'].value_counts()
        valid_countries = country_counts[country_counts >= min_samples].index
        df = df[df['country'].isin(valid_countries)]
        
        # Balance dataset if target_samples specified
        if target_samples:
            balanced_df = pd.DataFrame()
            for country in df['country'].unique():
                country_data = df[df['country'] == country]
                if len(country_data) > target_samples:
                    country_data = country_data.sample(n=target_samples, random_state=42)
                balanced_df = pd.concat([balanced_df, country_data])
            df = balanced_df.reset_index(drop=True)
            
        return df
    
def filter_and_kinda_balance_dataset(self, df, min_samples=30, target_samples=None):
        """
        Filter countries with minimum samples and optionally balance classes
        
        Args:
            df (pd.DataFrame): DataFrame with audio_path and country columns
            min_samples (int): Minimum samples required per country
            target_samples (int): Target number of samples per country for balancing
            
        Returns:
            pd.DataFrame: Filtered and balanced DataFrame
        """
        # Remove error/unknown countries
        df = df[~df['country'].isin(['Error', 'Unknown', 'Timeout'])]
        
        # Filter countries with minimum samples
        country_counts = df['country'].value_counts()
        valid_countries = country_counts[country_counts >= min_samples].index
        df = df[df['country'].isin(valid_countries)]
        
        # Balance dataset if target_samples specified
        if target_samples:
            balanced_df = pd.DataFrame()
            for country in df['country'].unique():
                country_data = df[df['country'] == country]
                if len(country_data) > target_samples:
                    country_data = country_data.sample(n=target_samples, random_state=42)
                balanced_df = pd.concat([balanced_df, country_data])
            df = balanced_df.reset_index(drop=True)
            
        return df

def main():
    parser = AudioLocationParser()
    artist_location_dict, audio_paths = parser.get_coordinates_with_audio_paths('/projects/dsci410_510/Kolahi_data_temp/fma_metadata/tracks.csv')
    paths_w_countries = parser.parse_locations(artist_location_dict, save_path='/projects/dsci410_510/Kolahi_data_temp/medium_tracks_with_countries.csv')
    #balanced_df = parser.filter_and_balance_dataset(paths_w_countries, min_samples=30, target_samples=30)
    #balanced_df.to_csv('/projects/dsci410_510/Kolahi_data_temp/tracks_with_countries.csv', index=False)
    print(paths_w_countries.head())
    print(len(paths_w_countries))
    print(paths_w_countries['country'].value_counts())

    #artist_location_dict, audio_paths = parser.get_coordinates_with_audio_paths('/projects/dsci410_510/Kolahi_data_temp/fma_metadata/tracks.csv')
    #paths_w_countries = parser.parse_locations(artist_location_dict, save_path='/projects/dsci410_510/Kolahi_data_temp/temp_tracks_with_countries.csv')
    #balanced_df = parser.filter_and_balance_dataset(paths_w_countries, min_samples=30, target_samples=30)
    #balanced_df.to_csv('/projects/dsci410_510/Kolahi_data_temp/tracks_with_countries.csv', index=False)

if __name__ == "__main__":
    main()