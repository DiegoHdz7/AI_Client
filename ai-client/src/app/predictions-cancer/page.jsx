'use client';
import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Table } from 'react-bootstrap';
import axios from 'axios';
import CustomNavbar from '../Navbar';

const ResponseKeysTable = () => {
  const [responseData, setResponseData] = useState([]);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/get-all-breast-cancer');
        console.log(response.data);
        setResponseData(response.data.all_breast_cancer);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  const flattenObject = (obj, parentKey = '') => {
    return Object.entries(obj).reduce((acc, [key, value]) => {
      if (typeof value === 'object' && value !== null) {
        acc.push(...flattenObject(value, `${parentKey}${key}.`));
      } else {
        acc.push([`${parentKey}${key}`, value]);
      }
      return acc;
    }, []);
  };

  const allRows = responseData.map((data, index) => {
    const flattenedData = flattenObject(data);
    const row = {};
    flattenedData.forEach(([key, value]) => {
      row[key] = value;
    });
    return { ...row, id: index + 1 };
  });

  const allKeys = Array.from(new Set(allRows.flatMap(Object.keys)));

  return (
    <>
      <CustomNavbar/>
      <Container>
        <Row>
          <Col>
            <h2>Response Data</h2>
            <Table striped bordered hover>
              <thead>
                <tr>
                  <th>#</th>
                  {allKeys.map((key, index) => (
                    <th key={index}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {allRows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    <td>{rowIndex + 1}</td>
                    {allKeys.map((key, keyIndex) => (
                      <td key={keyIndex}>{row[key]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </Table>
          </Col>
        </Row>
      </Container>
    </>
  );
};

export default ResponseKeysTable;