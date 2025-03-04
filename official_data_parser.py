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


    def get_coordinates_with_audio_paths(self, audio_path):
        tracks = utils.load('/projects/dsci410_510/Kolahi_dataset/audio_data/fma_small/tracks.csv')
        small = tracks[tracks['set', 'subset'] <= 'small']
        
        artist_location_dict = {}
        for track_id, artist in small['artist'].iterrows():
            lat = artist['latitude']
            lon = artist['longitude']
            if pd.notna(lat) and pd.notna(lon):
                artist_location_dict[track_id] = (lat, lon)

        audio_paths = []
        for track_id, coordinates in artist_location_dict.items():
            audio_path = utils.get_audio_path('/Users/cyruskolahi/Documents/372:410 VENV/DL project/data/fma_small', track_id)
            audio_paths.append(audio_path)
        
        return artist_location_dict


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

    def parse_locations(self, coordinates_file, save_path=None, save_interval=10):
        """
        Parse coordinates file and get corresponding countries
        
        Args:
            coordinates_file (str): Path to CSV with latitude/longitude coordinates
            save_path (str): Path to save output CSV
            save_interval (int): How often to save progress
            
        Returns:
            pd.DataFrame: DataFrame with audio paths and countries
        """
        # Load coordinates
        df = pd.read_csv(coordinates_file)
        countries = []
        
        for i in range(len(df)):
            latitude = df['latitude'][i]
            longitude = df['longitude'][i]
            print(f"Processing {i+1}/{len(df)}: {latitude}, {longitude}")
            
            country = self.get_country_from_coordinates(latitude, longitude)
            countries.append(country)
            
            # Add delay between requests
            time.sleep(1)
            
            # Save progress periodically
            if save_path and (i + 1) % save_interval == 0:
                temp_df = df.copy()
                temp_df['country'] = countries + [''] * (len(df) - len(countries))
                temp_df.to_csv(save_path.replace('.csv', '_partial.csv'), index=False)

        # Create final dataframe
        df['country'] = countries
        
        # Save final results if path provided
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nCompleted! Results saved to {save_path}")
            print(f"Successfully processed {len([c for c in countries if c != 'Timeout' and c != 'Error'])} locations")

        return df[['audio_path', 'country']]

    def filter_and_balance_dataset(self, df, min_samples=10, target_samples=None):
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
    parser.parse_locations('/projects/dsci410_510/Kolahi_dataset/audio_data/fma_small/tracks.csv', save_path='/projects/dsci410_510/Kolahi_dataset/audio_data/fma_small/tracks_with_countries.csv')
    parser.filter_and_balance_dataset('/projects/dsci410_510/Kolahi_dataset/audio_data/fma_small/tracks_with_countries.csv', min_samples=10, target_samples=10)
if __name__ == "__main__":
    main()