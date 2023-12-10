import Image from 'next/image'
import styles from './page.module.css'
import { Container, Row, Button, Col } from "react-bootstrap"
import CustomNavbar from './Navbar'

export default function Home() {
  return (
    <>
    <CustomNavbar/>
    <Container fluid>
      <Row className="justify-content-center align-items-center text-center">
        <h1>Welcome to Ai Diseases prediction</h1>
      </Row>
      <Row className="justify-content-center align-items-center text-center">
        <p>
          We created a website to predict whether a patient is diagnosed  with an specific disease
        </p>
      </Row>
      <Row className="justify-content-center align-items-center">
      <Col xs={12} sm={6} lg={4} className="text-center">
       
          <Button>Predict</Button>
        </Col>

      </Row>
    </Container>
    </>
  )
}
