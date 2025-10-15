import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
parent_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_model_dir)
parent_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
sys.path.append(parent_data_dir)
from content_based_with_model import (
    preprocess_audio,
    extract_vggish_embedding,
    get_recommendations,
    get_recommendations_names,
    get_playlist_recommendations,
    get_playlist_recommendations_names,
    _get_name_artist_pairs,
    RecInput,
    songs_metadata,
    TRACKS_EMBEDDINGS_SIMILARS_PATH,
    TRACKS_FILES_PATH,
)

class TestContentBasedWithModel(unittest.TestCase):
    def setUp(self):
        # Mock songs_metadata DataFrame
        self.mock_songs_metadata = pd.DataFrame({
            'spotify_id': ['track1', 'track2', 'track3'],
            'name': ['Song 1', 'Song 2', 'Song 3'],
            'artist': ['Artist 1', 'Artist 2', 'Artist 3'],
            'tags': [['pop', 'rock'], ['jazz'], ['pop']]
        })
        self.patcher_songs_metadata = patch(
            'content_based_with_model.songs_metadata',
            self.mock_songs_metadata
        )
        self.patcher_songs_metadata.start()

        # Mock VGGish model
        self.mock_vggish_model = MagicMock()
        self.patcher_vggish_model = patch(
            'content_based_with_model.vggish_model',
            self.mock_vggish_model
        )
        self.patcher_vggish_model.start()

        # Mock environment variables
        self.patcher_os = patch('content_based_with_model.os')
        self.mock_os = self.patcher_os.start()
        self.mock_os.getenv.side_effect = lambda x, y: y

        # Mock Path objects
        self.mock_base_dir = Path('/mock/path')
        self.patcher_path = patch('content_based_with_model.Path')
        self.mock_path = self.patcher_path.start()
        self.mock_path.return_value.resolve.return_value.parent = self.mock_base_dir
        self.mock_path.return_value.__truediv__ = MagicMock(return_value=self.mock_base_dir)

    def tearDown(self):
        self.patcher_songs_metadata.stop()
        self.patcher_vggish_model.stop()
        self.patcher_os.stop()
        self.patcher_path.stop()

    # ---------------------------------------------------------------------- #
    # 1) Tests successful audio preprocessing with valid input
    # ---------------------------------------------------------------------- #
    @patch('content_based_with_model.librosa.load')
    def test_preprocess_audio_success(self, mock_librosa_load):
        # Arrange
        mock_librosa_load.return_value = (np.zeros(16000), 16000)  # 1-second audio
        file_path = 'test.mp3'

        # Act
        result = preprocess_audio(file_path)

        # Assert
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 16000)
        mock_librosa_load.assert_called_once_with(file_path, sr=16000, mono=True)

    # ---------------------------------------------------------------------- #
    # 2) Tests audio preprocessing with short audio that needs padding
    # ---------------------------------------------------------------------- #
    @patch('content_based_with_model.librosa.load')
    def test_preprocess_audio_short_audio(self, mock_librosa_load):
        # Arrange
        mock_librosa_load.return_value = (np.zeros(8000), 16000)  # 0.5-second audio
        file_path = 'test.mp3'

        # Act
        result = preprocess_audio(file_path)

        # Assert
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), int(0.96 * 16000))  # Should be padded
        mock_librosa_load.assert_called_once_with(file_path, sr=16000, mono=True)

    # ---------------------------------------------------------------------- #
    # 3) Tests audio preprocessing failure handling
    # ---------------------------------------------------------------------- #
    @patch('content_based_with_model.librosa.load')
    def test_preprocess_audio_failure(self, mock_librosa_load):
        # Arrange
        mock_librosa_load.side_effect = Exception("Load error")
        file_path = 'test.mp3'

        # Act
        result = preprocess_audio(file_path)

        # Assert
        self.assertIsNone(result)
        mock_librosa_load.assert_called_once_with(file_path, sr=16000, mono=True)

    # ---------------------------------------------------------------------- #
    # 4) Tests successful VGGish embedding extraction
    # ---------------------------------------------------------------------- #
    def test_extract_vggish_embedding_success(self):
        # Arrange
        audio = np.zeros(int(0.96 * 16000))
        self.mock_vggish_model.return_value = np.ones((10, 128))  # Mock embeddings

        # Act
        result = extract_vggish_embedding(audio)

        # Assert
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (128,))
        np.testing.assert_array_equal(result, np.ones(128))

    # ---------------------------------------------------------------------- #
    # 5) Tests VGGish embedding extraction failure handling
    # ---------------------------------------------------------------------- #
    def test_extract_vggish_embedding_failure(self):
        # Arrange
        audio = np.zeros(int(0.96 * 16000))
        self.mock_vggish_model.side_effect = Exception("Embedding error")

        # Act
        result = extract_vggish_embedding(audio)

        # Assert
        self.assertIsNone(result)

    # ---------------------------------------------------------------------- #
    # 6) Tests recommendations retrieval for existing track
    # ---------------------------------------------------------------------- #
    @patch('content_based_with_model.pickle')
    @patch('content_based_with_model.preprocess_audio')
    @patch('content_based_with_model.extract_vggish_embedding')
    @patch('content_based_with_model.cosine_similarity')
    def test_get_recommendations_existing(self, mock_cosine, mock_extract, mock_preprocess, mock_pickle):
        # Arrange
        input_data = RecInput(spotify_id='track1', tags=['pop'])
        mock_pickle.load.return_value = [
            ('track1', np.ones(128), ['track2', 'track3']),
            ('track2', np.zeros(128), ['track1', 'track3']),
        ]
        mock_open_builtin = mock_open()
        with patch('builtins.open', mock_open_builtin):
            # Act
            result = get_recommendations(input_data)

        # Assert
        self.assertEqual(result, ['track2', 'track3'])
        mock_pickle.load.assert_called_once()
        mock_preprocess.assert_not_called()
        mock_extract.assert_not_called()
        mock_cosine.assert_not_called()

    # ---------------------------------------------------------------------- #
    # 7) Tests recommendations for new track with audio processing
    # ---------------------------------------------------------------------- #
    @patch('content_based_with_model.pickle')
    @patch('content_based_with_model.preprocess_audio')
    @patch('content_based_with_model.extract_vggish_embedding')
    @patch('content_based_with_model.cosine_similarity')
    def test_get_recommendations_new_track(self, mock_cosine, mock_extract, mock_preprocess, mock_pickle):
        # Arrange
        input_data = RecInput(spotify_id='new_track', tags=['pop'])
        mock_pickle.load.return_value = [
            ('track1', np.ones(128), ['track2', 'track3']),
            ('track2', np.zeros(128), ['track1', 'track3']),
        ]
        mock_preprocess.return_value = np.zeros(int(0.96 * 16000))
        mock_extract.return_value = np.ones(128)
        mock_cosine.return_value = np.array([[0.9, 0.1]])
        mock_open_builtin = mock_open()
        with patch('builtins.open', mock_open_builtin):
            # Act
            result = get_recommendations(input_data)

        # Assert
        self.assertEqual(result, ['track1', 'track2'])
        mock_preprocess.assert_called_once_with(TRACKS_FILES_PATH / 'new_track.mp3')
        mock_extract.assert_called_once()
        mock_cosine.assert_called_once()
        mock_pickle.dump.assert_called_once()

    # ---------------------------------------------------------------------- #
    # 8) Tests recommendations failure due to audio preprocessing error
    # ---------------------------------------------------------------------- #
    @patch('content_based_with_model.preprocess_audio')
    def test_get_recommendations_audio_failure(self, mock_preprocess):
        # Arrange
        input_data = RecInput(spotify_id='track1', tags=['pop'])
        mock_preprocess.return_value = None

        # Act
        result = get_recommendations(input_data)

        # Assert
        self.assertEqual(result, [])

    # ---------------------------------------------------------------------- #
    # 9) Tests retrieval of song names and artists for recommendations
    # ---------------------------------------------------------------------- #
    def test_get_recommendations_names(self):
        # Arrange
        input_data = RecInput(spotify_id='track1', tags=['pop'])
        with patch('content_based_with_model.get_recommendations', return_value=['track2', 'track3']):
            # Act
            result = get_recommendations_names(input_data)

        # Assert
        self.assertEqual(result, [('Song 2', 'Artist 2'), ('Song 3', 'Artist 3')])

    # ---------------------------------------------------------------------- #
    # 10) Tests playlist recommendations with valid input
    # ---------------------------------------------------------------------- #
    def test_get_playlist_recommendations_valid(self):
        # Arrange
        seed_ids = ['track1', 'track2', 'track3', 'track4', 'track5']
        m = 10
        mock_recommendations = [
            ['track3', 'track2', 'other1', 'other2'],  # for track1
            ['track1', 'track3', 'other3', 'other4'],  # for track2
            ['track1', 'track2', 'other5', 'other6'],  # for track3
            ['track1', 'track2', 'other7', 'other8'],  # for track4
            ['track1', 'track2', 'other9', 'other10'],  # for track5
        ] * 4  # Enough for multiple rounds
        with patch(
            'content_based_with_model.get_recommendations',
            side_effect=mock_recommendations
        ):
            # Act
            result = get_playlist_recommendations(seed_ids, m)

        # Assert
        self.assertEqual(len(result), 10)
        self.assertEqual(len(set(result)), 10)  # Ensure uniqueness
        self.assertTrue(all(sid not in seed_ids for sid in result))  # No seeds in result

    # ---------------------------------------------------------------------- #
    # 11) Tests playlist recommendations with insufficient seed tracks
    # ------------------------------------------------------------------- #
    def test_get_playlist_recommendations_too_few_seeds(self):
        # Arrange
        seed_ids = ['track1', 'track2']
        m = 10

        # Act/Assert
        with self.assertRaises(ValueError) as cm:
            get_playlist_recommendations(seed_ids, m)
        self.assertEqual(str(cm.exception), "Need at least 5 seed tracks")

    # ---------------------------------------------------------------------- #
    # 12) Tests playlist recommendations with invalid m value
    # ---------------------------------------------------------------------- #
    def test_get_playlist_recommendations_invalid_m(self):
        # Arrange
        seed_ids = ['track1', 'track2', 'track3', 'track4', 'track5']
        m = 5

        # Act/Assert
        with self.assertRaises(ValueError) as cm:
            get_playlist_recommendations(seed_ids, m)
        self.assertEqual(str(cm.exception), "m must satisfy n < m ≤ 25")

    # ---------------------------------------------------------------------- #
    # 13) Tests playlist recommendations when no recommendations are generated
    # ---------------------------------------------------------------------- #
    def test_get_playlist_recommendations_no_recs(self):
        # Arrange
        seed_ids = ['track1', 'track2', 'track3', 'track4', 'track5']
        m = 10
        with patch('content_based_with_model.get_recommendations', return_value=[]):
            # Act/Assert
            with self.assertRaises(ValueError) as cm:
                get_playlist_recommendations(seed_ids, m)
            self.assertEqual(str(cm.exception), "Failed to generate recommendations for spotify_id track1")

    # ---------------------------------------------------------------------- #
    # 14) Tests retrieval of song names and artists for playlist recommendations
    # ---------------------------------------------------------------------- #
    def test_get_playlist_recommendations_names(self):
        # Arrange
        seed_ids = ['track1', 'track2', 'track3', 'track4', 'track5']
        m = 10
        with patch(
            'content_based_with_model.get_playlist_recommendations',
            return_value=['track2', 'track3']
        ):
            # Act
            result = get_playlist_recommendations_names(seed_ids, m)

        # Assert
        self.assertEqual(result, [('Song 2', 'Artist 2'), ('Song 3', 'Artist 3')])

    # ---------------------------------------------------------------------- #
    # 15) Tests conversion of Spotify IDs to song name and artist pairs
    # ---------------------------------------------------------------------- #
    def test_get_name_artist_pairs(self):
        # Arrange
        spotify_ids = ['track1', 'track2', 'invalid_track']

        # Act
        result = _get_name_artist_pairs(spotify_ids)

        # Assert
        self.assertEqual(result, [('Song 1', 'Artist 1'), ('Song 2', 'Artist 2')])

if __name__ == '__main__':
    unittest.main()