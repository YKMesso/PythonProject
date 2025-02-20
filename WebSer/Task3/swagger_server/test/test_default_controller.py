# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_example_get(self):
        """Test case for example_get

        
        """
        response = self.client.open(
            '/v1/example',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_all_prod_get(self):
        """Test case for get_all_prod_get

        
        """
        response = self.client.open(
            '/v1/getAllProd',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_single_product_get(self):
        """Test case for get_single_product_get

        
        """
        response = self.client.open(
            '/v1/getSingleProduct',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_insert_new_product_get(self):
        """Test case for insert_new_product_get

        
        """
        response = self.client.open(
            '/v1/insertNewProduct',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_search_get(self):
        """Test case for search_get

        
        """
        response = self.client.open(
            '/v1/search',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_test_get(self):
        """Test case for test_get

        
        """
        response = self.client.open(
            '/v1/test',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
