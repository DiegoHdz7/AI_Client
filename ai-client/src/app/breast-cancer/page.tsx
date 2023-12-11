"use client";
import { useState,useEffect, ChangeEvent, FormEvent } from "react";
import { Form, Button } from "react-bootstrap";
import CustomNavbar from "@/app/Navbar";
import Badge from "react-bootstrap/Badge";
import axios from 'axios';

export default function BreastCancerDiagnosis() {
  const [prediction,setPrediction] = useState({label:null,predictions:[]});

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    
    const formData = new FormData(document.forms[0]);
    const modelInput: Record<string, string> = {};
    

    formData.forEach((value, name) => {
      modelInput[name] = value.toString();
    });
    console.log('Body',modelInput)

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict/breast-cancer',modelInput);
  
      // Access the prediction and predictions from the response data
       setPrediction({label:response.data.prediction,predictions:response.data.predictions})
  
    } catch (error) {
      console.error('Error fetching data:', error);
    }


  };

 useEffect(()=>{
  console.log('prediction',prediction)

 },[prediction])

  return (
    <div>
      <CustomNavbar />
      <h1 className="justify-content-center align-items-center text-center">
        Diagnose
      </h1>
      <div className="d-flex justify-content-center align-items-center text-center">
        <Form
          style={{ maxWidth: "600px", width: "100%" }}
          onSubmit={handleSubmit}
          
        >
          <Form.Group controlId="formDishName">
            <Form.Label>Radius 1</Form.Label>
            <Form.Control
              type="number"
              placeholder="Enter Radius 1"
              name="radius1"
              required
              
            />
            <Form.Label>Perimeter 1</Form.Label>
            <Form.Control
              type="number"
              placeholder="Enter Perimeter 1"
              name="perimeter1"
              required
              
            />
            <Form.Label>Area 1</Form.Label>
            <Form.Control
              type="number"
              placeholder="Enter Area 1"
              name="area1"
              required
              
            />

            <Form.Label>Concavity 1</Form.Label>
            <Form.Control
              type="number"
              placeholder="Concavity 1"
              name="concavity1"
              required
              
            />

            <Form.Label>Concave points 1</Form.Label>
            <Form.Control
              type="number"
              placeholder="Concave points 1"
              name="concave_points1"
              required
              
            />
            <Form.Label>Radius 3</Form.Label>
            <Form.Control type="number" placeholder="Radius 3" name="radius3" />
            <Form.Label>Perimeter 3</Form.Label>
            <Form.Control
              type="number"
              placeholder="Perimeter 3"
              name="perimeter3"
              required
              
            />
            <Form.Label>Area 3</Form.Label>
            <Form.Control type="number" placeholder="Area 3" name="area3" />
            <Form.Label>Concavity 3</Form.Label>
            <Form.Control
              type="number"
              placeholder="Concavity 3"
              name="concavity3"
              required
              
            />
            <Form.Label>Concave points 3</Form.Label>
            <Form.Control
              type="number"
              placeholder="Concave points 3"
              name="concave_points3"
              required
              
            />
              <Button variant="primary" type="submit" onClick={handleSubmit}>
            Predict
          </Button>
          </Form.Group>

        
        </Form>
        {prediction.label != null ? (
          <div>
           <h4>
           Diagnose: <Badge bg="secondary">{prediction.label}</Badge> 
          </h4>
          <p>Benign Class:{(prediction.predictions[0][0]*100).toFixed(2)} <br/> Malign Class {(prediction.predictions[0][1]*100).toFixed(2)}</p>
          </div>
         
        ) : null}

        <></>
      </div>
    </div>
  );
}
