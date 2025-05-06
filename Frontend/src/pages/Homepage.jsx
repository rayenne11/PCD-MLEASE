import { Code, RefreshCw, Shield, BarChart3, Pointer } from "lucide-react";
import React, { useState } from "react";
import NavbarHome from "../components/NavbarHome";
import AIphoto from "../assets/background_ai.png";
import mlease_logo from "../assets/logo.svg";
import RotatingFront from "../assets/rotating-card-bg-front.jpeg";
import RotatingBack from "../assets/rotating-card-bg-back.jpeg";
import { motion } from "framer-motion";
import "./Homepage.css";

const HomePage = () => {
  return (
    <>
      <NavbarHome />
      <MLeaseLanding />
      <HeroSection />
      <FeaturesSection />
      <PageShowcase />
    </>
  );
};

export default HomePage;

function MLeaseLanding() {
  return (
    <div className="mlease-landing">
      <svg
        className="mlease-landing-svg"
        viewBox="0 0 1440 320"
        preserveAspectRatio="none"
      >
        <path
          fill="#ffffff"
          fillOpacity="0.3"
          d="M0,160L60,154.7C120,149,240,139,360,160C480,181,600,235,720,250.7C840,267,960,245,1080,229.3C1200,213,1320,203,1380,197.3L1440,192L1440,320L1380,320C1320,320,1200,320,1080,320C960,320,840,320,720,320C600,320,480,320,360,320C240,320,120,320,60,320L0,320Z"
        ></path>
      </svg>

      <div className="mlease-landing-content">
        <div className="mlease-landing-text">
          <h1 className="mlease-landing-title">
            Getting Started with MLEASE!
          </h1>
          <p className="mlease-landing-description">
            MLEASE is an MLOps-driven platform that simplifies and democratizes
            machine learning model deployment.
          </p>
          <button className="mlease-landing-button">
            <a href="/signin">Get Started Free</a>
          </button>
        </div>

        <div className="mlease-landing-image-container">
          <img src={AIphoto} alt="" />
        </div>
      </div>
    </div>
  );
}

function HeroSection() {
  return (
    <div className="hero-section">
      {/* Left side: Text content */}
      <div className="hero-text">
        <h1 className="hero-title">
          Simplify AI Operations with MLease
        </h1>
        <div>
          <span className="hero-highlight">MLEASE </span>
          <p className="hero-description">
            empowers users at all enterprise levels to manage, monitor, and
            operationalize ML models effortlessly. With intelligent automation,
            robust monitoring, and accessible tools, MLease bridges the gap
            between advanced AI technologies and practical, everyday
            implementation.
          </p>
        </div>
        <button className="hero-button">
          Get Started →
        </button>
      </div>

      {/* Right side: Logo */}
      <div className="hero-logo-container">
        <img src={mlease_logo} alt="MLease Logo" className="hero-logo" />
      </div>
    </div>
  );
}

function FeaturesSection() {
  const [flippedCard, setFlippedCard] = useState(false);

  const features = [
    {
      id: "integration",
      icon: <Code className="feature-icon" />,
      title: "Seamless Integration",
      description:
        "Connect and automate your ML workflows effortlessly with robust APIs and an intuitive dashboard.",
    },
    {
      id: "analytics",
      icon: <BarChart3 className="feature-icon" />,
      title: "Real-time Analytics",
      description:
        "Monitor your model performance live and make informed adjustments on the fly.",
    },
    {
      id: "pipelines",
      icon: <RefreshCw className="feature-icon" />,
      title: "Automated Pipelines",
      description:
        "Accelerate your deployment with end-to-end automation that takes your models from training to production seamlessly.",
    },
    {
      id: "security",
      icon: <Shield className="feature-icon" />,
      title: "Enterprise-Grade Security",
      description:
        "Safeguard your data and models with advanced encryption and comprehensive security features.",
    },
  ];

  return (
    <div className="features-section">
      {/* Left side: Rotating cards */}
      <div className="features-card-container">
        <div className="features-card-wrapper">
          <div className="features-card">
            {/* Front Side */}
            <div
              style={{ backgroundImage: `url(${RotatingFront})` }}
              className="features-card-front"
            >
              <div className="features-card-front-overlay" />
              <div className="features-card-content">
                <Pointer className="features-card-icon" size={50} />
                <h2 className="features-card-title">Experience MLease</h2>
                <p className="features-card-description">
                  Streamline your ML pipeline with a unified platform designed
                  for efficient model deployment.
                </p>
              </div>
            </div>

            {/* Back Side */}
            <div
              style={{ backgroundImage: `url(${RotatingBack})` }}
              className="features-card-back"
            >
              <div className="features-card-back-overlay" />
              <div className="features-card-content">
                <h2 className="features-card-title">Discover MLease</h2>
                <p className="features-card-description">
                  Dive into our intuitive dashboard, access real-time analytics,
                  and unlock end-to-end automation for your ML models.
                </p>
                <button className="features-card-button">
                  Explore Features
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Right side: Features grid */}
      <div className="features-grid">
        {features.map((feature) => (
          <div key={feature.id} className="feature-item">
            <div className="feature-icon-container">
              {feature.icon}
            </div>
            <h3 className="feature-title">
              {feature.title}
            </h3>
            <p className="feature-description">{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PageShowcase({
  aboutUsImage,
  contactUsImage,
  signInImage,
  authorImage,
}) {
  return (
    <div className="page-showcase">
      <div className="page-showcase-container">
        <div className="page-showcase-content">
          {/* Left side: Grid of page previews */}
          <div className="page-showcase-grid">
            {/* About Us Page */}
            <div className="page-showcase-item">
              <div className="page-showcase-image-container">
                {aboutUsImage ? (
                  <img
                    src={aboutUsImage}
                    alt="About Us Page Preview"
                    className="page-showcase-image"
                  />
                ) : (
                  <div className="page-showcase-image-placeholder">
                    <span className="page-showcase-image-placeholder-text">About Us Image</span>
                  </div>
                )}
              </div>
              <h3 className="page-showcase-title">
                About Us Page
              </h3>
            </div>

            {/* Contact Us Page */}
            <div className="page-showcase-item">
              <div className="page-showcase-image-container">
                {contactUsImage ? (
                  <img
                    src={contactUsImage}
                    alt="Contact Us Page Preview"
                    className="page-showcase-image"
                  />
                ) : (
                  <div className="page-showcase-image-placeholder">
                    <span className="page-showcase-image-placeholder-text">Contact Us Image</span>
                  </div>
                )}
              </div>
              <h3 className="page-showcase-title">
                Contact Us Page
              </h3>
            </div>

            {/* Sign In Page */}
            <div className="page-showcase-item">
              <div className="page-showcase-image-container">
                {signInImage ? (
                  <img
                    src={signInImage}
                    alt="Sign In Page Preview"
                    className="page-showcase-image"
                  />
                ) : (
                  <div className="page-showcase-image-placeholder">
                    <span className="page-showcase-image-placeholder-text">Sign In Image</span>
                  </div>
                )}
              </div>
              <h3 className="page-showcase-title">
                Sign In Page
              </h3>
            </div>

            {/* Author Page */}
            <div className="page-showcase-item">
              <div className="page-showcase-image-container">
                {authorImage ? (
                  <img
                    src={authorImage}
                    alt="Author Page Preview"
                    className="page-showcase-image"
                  />
                ) : (
                  <div className="page-showcase-image-placeholder">
                    <span className="page-showcase-image-placeholder-text">Author Image</span>
                  </div>
                )}
              </div>
              <h3 className="page-showcase-title">Author Page</h3>
            </div>
          </div>

          {/* Right side: Description */}
          <div className="page-showcase-description">
            <h2 className="page-showcase-description-title">
              Empowering MLOps, Simplified
            </h2>
            <p className="page-showcase-description-text">
              MLease empowers users of all expertise levels to manage, monitor,
              and operationalize ML models effectively with automation,
              end-to-end support, and accessible tools—bridging the gap between
              advanced ML technologies and practical implementation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}