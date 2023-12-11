'use client'
import { useState,useEffect, ChangeEvent, FormEvent } from "react";
import { Form, Button } from "react-bootstrap";
import CustomNavbar from "@/app/Navbar";
import { useRouter, useSearchParams } from "next/navigation";
import Badge from "react-bootstrap/Badge";
import axios from 'axios';

export default function TumorGradePrediction() {
  const [data, setData] = useState(null);
  const [prediction,setPrediction] = useState({label:null,predictions:[]});
  const router = useRouter();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    
    const formData = new FormData(document.forms[0]);
    const modelInput: Record<string, string> = {};
    

    formData.forEach((value, name) => {
      modelInput[name] = value.toString();
    });
    console.log('Body',modelInput)

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict/tumors',modelInput);
  
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
        Diagnose Grade of Tumour
      </h1>
      <div className="d-flex justify-content-center align-items-center text-center">
        <Form
          style={{ maxWidth: "600px", width: "100%" }}
          onSubmit={handleSubmit}
          
        >
          <Form.Group controlId="formTumor">
            <Form.Label>Age at Diagnosis</Form.Label>
            <Form.Control
              type="number"
              placeholder="Enter age"
              name="Age_at_diagnosis"
              required
              
            />
            <Form.Label>Race</Form.Label>
            <br></br>
            <Form.Label>0 = White, 1 = Black/African American, 2 = Asian</Form.Label>
            <Form.Control
              type="number"
              placeholder="Enter race value"
              name="Race"
              required
              
            />
            <Form.Label>Enter 1 for Mutated, 0 for Not Mutated</Form.Label>
            <br></br>
            <Form.Label>IDH1</Form.Label>
            <Form.Control
              type="number"
              placeholder="Enter IDH1"
              name="IDH1"
              required
              
            />

            <Form.Label>TP53</Form.Label>
            <Form.Control
              type="number"
              placeholder="TP53"
              name="TP53"
              required
              
            />

            <Form.Label>ATRX</Form.Label>
            <Form.Control
              type="number"
              placeholder="ATRX"
              name="ATRX"
              required
              
            />
            <Form.Label>PTEN</Form.Label>
            <Form.Control
              type="number"
              placeholder="PTEN"
              name="PTEN"
              required
              
            />
            <Form.Label>EGFR</Form.Label>
            <Form.Control
              type="number"
              placeholder="EGFR"
              name="EGFR"
              required
              
            />
            <Form.Label>CIC</Form.Label>
            <Form.Control
              type="number"
              placeholder="CIC"
              name="CIC"
              required
              
            />
            <Form.Label>MUC16</Form.Label>
            <Form.Control
              type="number"
              placeholder="MUC16"
              name="MUC16"
              required
              
            />
            <Form.Label>PIK3CA</Form.Label>
            <Form.Control
              type="number"
              placeholder="PIK3CA"
              name="PIK3CA"
              required
              
            />
            <Form.Label>NF1</Form.Label>
            <Form.Control
              type="number"
              placeholder="NF1"
              name="NF1"
              required
              
            />
            <Form.Label>PIK3R1</Form.Label>
            <Form.Control
              type="number"
              placeholder="PIK3R1"
              name="PIK3R1"
              required
              
            />
            <Form.Label>FUBP1</Form.Label>
            <Form.Control
              type="number"
              placeholder="FUBP1"
              name="FUBP1"
              required
              
            />
            <Form.Label>RB1</Form.Label>
            <Form.Control
              type="number"
              placeholder="RB1"
              name="RB1"
              required
              
            />
            <Form.Label>NOTCH1</Form.Label>
            <Form.Control
              type="number"
              placeholder="NOTCH1"
              name="NOTCH1"
              required
              
            />            
            <Form.Label>CSMD3</Form.Label>
            <Form.Control
              type="number"
              placeholder="CSMD3"
              name="CSMD3"
              required
              
            />            
            <Form.Label>SMARCA4</Form.Label>
            <Form.Control
              type="number"
              placeholder="SMARCA4"
              name="SMARCA4"
              required
              
            />            
            <Form.Label>GRIN2A</Form.Label>
            <Form.Control
              type="number"
              placeholder="GRIN2A"
              name="GRIN2A"
              required
              
            />            
            <Form.Label>IDH2</Form.Label>
            <Form.Control
              type="number"
              placeholder="IDH2"
              name="IDH2"
              required
              
            />            
            <Form.Label>PDGFRA</Form.Label>
            <Form.Control
              type="number"
              placeholder="PDGFRA"
              name="PDGFRA"
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
          <p>LGG Class:{(prediction.predictions[0][0]*100).toFixed(2)} <br/> GBM Class {(prediction.predictions[0][1]*100).toFixed(2)}</p>
          </div>
         
        ) : null}

        <></>
      </div>
    </div>
  );
}
