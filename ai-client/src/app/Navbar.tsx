'use client'
import { Navbar, Container, Nav, NavDropdown } from 'react-bootstrap';

const CustomNavbar: React.FC = () => {
    return (
        <Navbar expand="lg" className="bg-body-tertiary">
          <Container>
            <Navbar.Brand href="/">Restaurants Review</Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              <Nav className="me-auto">
                <Nav.Link href="/">Home</Nav.Link>
                <Nav.Link href="/breast-cancer">Breast Cancer Diagnosis</Nav.Link>
                <Nav.Link href="/tumor-grades">Tumor Grade Diagnosis</Nav.Link>
                <Nav.Link href="/predictions-cancer">Breast Cancer Predictions</Nav.Link>
                <Nav.Link href="/predictions-tumors">Tumor Grade Diagnosis Pred</Nav.Link>
              </Nav>
            </Navbar.Collapse>
          </Container>
        </Navbar>
      );
    }

export default CustomNavbar;