import unittest
import json
from app import app

# class FlaskTestCase(unittest.TestCase):

#     def setUp(self):
#         self.app = app.test_client()
#         self.app.testing = True

#     def test_predict(self):
#         # Example input data similar to what was used during training
#         sample_data = {
#             "o_rice": [500, 502, 505, 507, 510, 515, 518, 520, 523, 525, 527, 530],
#             "h_rice": [520, 523, 525, 527, 530, 532, 535, 538, 540, 543, 545, 548],
#             "l_rice": [480, 482, 485, 487, 490, 492, 495, 497, 500, 502, 505, 507],
#             "c_rice": [505, 507, 510, 512, 515, 518, 520, 523, 525, 528, 530, 533]
#         }

#         response = self.app.post('/predict', data=json.dumps(sample_data), content_type='application/json')

#         # Check for a successful response
#         self.assertEqual(response.status_code, 200)
#         # Check if the response is a JSON and contains 'predictions'
#         self.assertIn('predictions', response.get_json())

class TestApp(unittest.TestCase):

    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)
        
if __name__ == '__main__':
    unittest.main()
