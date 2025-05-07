import { Code, RefreshCw, Shield, BarChart3, Pointer } from "lucide-react";
import React, { useState } from "react";
import NavbarHome from "../components/NavbarHome";
import AIphoto from "../assets/background_ai.png";
import mlease_logo from "../assets/logo.svg";
import RotatingFront from "../assets/rotating-card-bg-front.jpeg";
import RotatingBack from "../assets/rotating-card-bg-back.jpeg";
import Training from "../assets/How-do-you-train-AI.jpg";
import EDA from "../assets/EDA.png";
import Tracking from "../assets/metrics-tracing-logging.png";
import Pipeline from "../assets/workflow-orchestration.png";
import { motion } from "framer-motion";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (delay = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay, duration: 0.6, ease: "easeOut" },
  }),
};

const HomePage = () => {
  return (
    <>
      <NavbarHome />
      <MLeaseLanding />
      <HeroSection />
      <FeaturesSection />
      <PageShowcase />
      <ContactUs />
      <Footer />
    </>
  );
};

export default HomePage;

function MLeaseLanding() {
  return (
    <motion.div
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      className="relative bg-gradient-to-br from-[#FD9D65] to-[#FFB88C] w-full h-main flex items-center justify-center p-6 overflow-hidden"
    >
      <svg
        className="absolute top-0 left-0 w-full h-full z-0 opacity-35"
        viewBox="0 0 1440 320"
        preserveAspectRatio="none"
      >
        <path
          fill="#ffffff"
          fillOpacity="0.3"
          d="M0,160L60,154.7C120,149,240,139,360,160C480,181,600,235,720,250.7C840,267,960,245,1080,229.3C1200,213,1320,203,1380,197.3L1440,192L1440,320L1380,320C1320,320,1200,320,1080,320C960,320,840,320,720,320C600,320,480,320,360,320C240,320,120,320,60,320L0,320Z"
        ></path>
      </svg>

      <motion.div
        className="relative z-10 max-w-6xl w-full flex flex-col md:flex-row items-center justify-between gap-8"
        variants={fadeUp}
      >
        <motion.div
          className="space-y-6 max-w-lg"
          custom={0.2}
          variants={fadeUp}
        >
          <h1 className="text-5xl font-bold text-white drop-shadow-md">
            Getting Started with MLEASE!
          </h1>
          <p className="text-xl text-white">
            MLEASE is an MLOps-driven platform that simplifies and democratizes
            machine learning model deployment.
          </p>
          <a href="/signin">
            <button className="bg-blue-500 hover:bg-blue-600 text-white py-3 px-6 rounded-md font-medium transition-colors flex items-center">
              Get Started Free
            </button>
          </a>
        </motion.div>

        <motion.div
          className="bg-white p-6 rounded-xl shadow-lg max-w-sm"
          custom={0.4}
          variants={fadeUp}
        >
          <img src={AIphoto} alt="AI" />
        </motion.div>
      </motion.div>
    </motion.div>
  );
}

function HeroSection() {
  return (
    <motion.div
      className="flex flex-col lg:flex-row items-center justify-around w-full h-main bg-white py-16 px-4 sm:px-6 lg:px-8"
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      variants={fadeUp}
    >
      <motion.div
        className="max-w-xl mb-10 lg:mb-0"
        custom={0.1}
        variants={fadeUp}
      >
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
          Simplify AI Operations with MLease
        </h1>
        <div className="mb-8">
          <span className="font-semibold text-gray-700">MLEASE </span>
          <p className="text-gray-600 mt-2 inline">
            empowers users at all enterprise levels to manage, monitor, and
            operationalize ML models effortlessly...
          </p>
        </div>
        <button className="bg-orange-300 hover:bg-orange-400 text-white py-2 px-6 rounded-md cursor-pointer transition-colors">
          Get Started →
        </button>
      </motion.div>

      <motion.div
        className="bg-white p-8 rounded-lg shadow-lg"
        custom={0.3}
        variants={fadeUp}
      >
        <img src={mlease_logo} alt="MLease Logo" className="w-64 h-auto" />
      </motion.div>
    </motion.div>
  );
}

function FeaturesSection() {
  const features = [
    {
      id: "integration",
      icon: <Code className="w-5 h-5 text-blue-600" />,
      title: "Seamless Integration",
      description: "Connect and automate your ML workflows...",
      bg: "bg-blue-100",
    },
    {
      id: "analytics",
      icon: <BarChart3 className="w-5 h-5 text-green-600" />,
      title: "Real-time Analytics",
      description: "Monitor your model performance live...",
      bg: "bg-green-100",
    },
    {
      id: "pipelines",
      icon: <RefreshCw className="w-5 h-5 text-amber-600" />,
      title: "Automated Pipelines",
      description: "Accelerate your deployment...",
      bg: "bg-amber-100",
    },
    {
      id: "security",
      icon: <Shield className="w-5 h-5 text-red-600" />,
      title: "Enterprise-Grade Security",
      description: "Safeguard your data and models...",
      bg: "bg-red-100",
    },
  ];

  return (
    <motion.div
      className="w-full h-main bg-gray-50 py-16 px-4 sm:px-6 lg:px-8 flex flex-col lg:flex-row items-center gap-x-24 justify-center"
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
    >
      <motion.div
        className="h-[65dvh] w-[50dvh] flex justify-center items-center rounded-2xl"
        variants={fadeUp}
      >
        <div className="group [perspective:1000px] h-full w-full rounded-2xl">
          <div className="relative rounded-2xl w-full h-full transition-transform duration-700 [transform-style:preserve-3d] group-hover:[transform:rotateY(180deg)]">
            <div
              style={{ backgroundImage: `url(${RotatingFront})` }}
              className="absolute inset-0 text-white rounded-2xl shadow-xl flex flex-col justify-center bg-cover bg-center items-center [backface-visibility:hidden]"
            >
              <div className="absolute inset-0 bg-[linear-gradient(195deg,rgba(73,163,241,0.85),rgba(73,163,241,0.85))] rounded-2xl z-0" />
              <div className="w-full h-full flex justify-center relative items-center flex-col">
                <Pointer className="mb-16" size={50} />
                <h2 className="text-2xl font-bold">Experience MLease</h2>
                <p className="mt-2 text-center px-6">
                  Streamline your ML pipeline...
                </p>
              </div>
            </div>
            <div
              style={{ backgroundImage: `url(${RotatingBack})` }}
              className="absolute inset-0 text-white rounded-2xl shadow-xl flex flex-col justify-center items-center [transform:rotateY(180deg)] bg-cover bg-center [backface-visibility:hidden]"
            >
              <div className="absolute inset-0 bg-[linear-gradient(195deg,rgba(73,163,241,0.85),rgba(73,163,241,0.85))] rounded-2xl z-0" />
              <div className="w-full h-full flex justify-center relative items-center flex-col">
                <h2 className="text-2xl font-bold">Discover MLease</h2>
                <p className="mt-2 text-center px-6">
                  Dive into our intuitive dashboard...
                </p>
                <a href="/dashboard">
                  <button className="mt-6 px-4 py-2 bg-white text-blue-600 font-semibold rounded-xl shadow hover:bg-gray-100 transition">
                    Explore Features
                  </button>
                </a>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="w-full lg:w-1/2 grid md:grid-cols-2 gap-6">
        {features.map((feature, idx) => (
          <motion.div
            key={feature.id}
            className="bg-white rounded-2xl shadow-md p-6 hover:shadow-lg transition"
            custom={idx * 0.2}
            variants={fadeUp}
          >
            <div
              className={`mb-4 w-10 h-10 rounded-full flex items-center justify-center ${feature.bg}`}
            >
              {feature.icon}
            </div>
            <h3 className="text-lg font-semibold text-gray-800 mb-1">
              {feature.title}
            </h3>
            <p className="text-gray-600 text-sm">{feature.description}</p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function PageShowcase() {
  const cards = [
    { label: "EDA & insights", image: EDA },
    { label: "Autonomous Training", image: Training },
    { label: "Experiment Tracking", image: Tracking },
    {
      label: (
        <>
          End-to-End <br />
          Pipeline Orchestration
        </>
      ),
      image: Pipeline,
    },
  ];

  return (
    <motion.div
      className="w-full h-main bg-gray-50 py-20 px-6 sm:px-8"
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      variants={fadeUp}
    >
      <div className="max-w-7xl mx-auto h-full">
        <div className="flex flex-col lg:flex-row items-start justify-between gap-16 h-full">
          <motion.div
            className="w-full lg:w-3/5 grid md:grid-cols-2 gap-8"
            variants={fadeUp}
          >
            {cards.map(({ label, image }, idx) => (
              <motion.div
                key={idx}
                className="flex flex-col"
                custom={idx * 0.2}
                variants={fadeUp}
              >
                <div className="rounded-2xl overflow-hidden shadow-md hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                  <img
                    src={image}
                    alt={`${label} Preview`}
                    className="w-full object-cover aspect-video"
                  />
                </div>
                <h3 className="mt-3 text-center text-lg font-semibold text-gray-800">
                  {label}
                </h3>
              </motion.div>
            ))}
          </motion.div>

          <motion.div
            className="w-full lg:w-2/5 flex flex-col justify-center h-full"
            custom={0.3}
            variants={fadeUp}
          >
            <h2 className="text-4xl md:text-5xl font-extrabold text-gray-900 leading-tight mb-6">
              Empowering MLOps, Simplified
            </h2>
            <p className="text-gray-600 text-lg leading-relaxed">
              MLease empowers users of all expertise levels...
            </p>
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
}

function ContactUs() {
  return (
    <div className="w-full bg-gray-50 py-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Get in Touch
        </h2>
        <p className="text-gray-600 text-center mb-10">
          Have a question, feedback, or partnership opportunity? We'd love to
          hear from you.
        </p>
        <form className="space-y-6">
          <div>
            <label
              htmlFor="name"
              className="block text-sm font-medium text-gray-700"
            >
              Name
            </label>
            <input
              type="text"
              id="name"
              name="name"
              className="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="Your full name"
              required
            />
          </div>

          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-gray-700"
            >
              Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              className="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="your@email.com"
              required
            />
          </div>

          <div>
            <label
              htmlFor="message"
              className="block text-sm font-medium text-gray-700"
            >
              Message
            </label>
            <textarea
              id="message"
              name="message"
              rows={4}
              className="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="What would you like to tell us?"
              required
            ></textarea>
          </div>

          <div className="text-center">
            <button
              type="submit"
              className="inline-block bg-indigo-600 text-white px-6 py-2 rounded-md hover:bg-indigo-700 transition"
            >
              Send Message
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-6 md:space-y-0">
          {/* Logo or brand name */}
          <div className="text-xl font-semibold text-gray-800">MLease</div>

          {/* Links */}
          <div className="flex space-x-6 text-sm text-gray-600">
            <a href="#features" className="hover:text-blue-600 transition">
              Features
            </a>
            <a href="#contact" className="hover:text-blue-600 transition">
              Contact
            </a>
            <a href="#demo" className="hover:text-blue-600 transition">
              Demo
            </a>
            <a href="#faq" className="hover:text-blue-600 transition">
              FAQ
            </a>
          </div>

          {/* Socials */}
          <div className="flex space-x-4">
            <a
              href="#"
              aria-label="GitHub"
              className="text-gray-500 hover:text-gray-800 transition"
            >
              <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 008.01 10.96c.58.1.79-.25.79-.56v-2.1c-3.26.71-3.95-1.55-3.95-1.55-.53-1.34-1.3-1.7-1.3-1.7-1.06-.73.08-.72.08-.72 1.17.08 1.78 1.21 1.78 1.21 1.04 1.77 2.72 1.26 3.38.97.1-.75.4-1.26.72-1.55-2.6-.3-5.33-1.3-5.33-5.8 0-1.28.46-2.33 1.21-3.15-.12-.3-.52-1.5.12-3.13 0 0 .98-.31 3.2 1.2A11.03 11.03 0 0112 5.5c.99.01 1.99.13 2.92.39 2.22-1.51 3.2-1.2 3.2-1.2.64 1.63.24 2.83.12 3.13.76.82 1.21 1.87 1.21 3.15 0 4.52-2.74 5.49-5.35 5.79.42.36.77 1.08.77 2.19v3.24c0 .31.21.67.8.56A11.5 11.5 0 0023.5 12C23.5 5.65 18.35.5 12 .5z" />
              </svg>
            </a>
            <a
              href="#"
              aria-label="LinkedIn"
              className="text-gray-500 hover:text-gray-800 transition"
            >
              <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M4.98 3.5C3.34 3.5 2 4.83 2 6.48s1.34 2.98 2.98 2.98 2.98-1.34 2.98-2.98S6.62 3.5 4.98 3.5zM2.4 21.5h5.16V8.98H2.4V21.5zM9.6 8.98V21.5h5.15v-6.68c0-3.6 4.55-3.89 4.55 0V21.5h5.15v-7.84c0-6.2-6.68-5.96-8.66-2.91v-1.77H9.6z" />
              </svg>
            </a>
          </div>
        </div>

        {/* Bottom Text */}
        <div className="mt-8 text-center text-sm text-gray-500">
          © {new Date().getFullYear()} MLease. All rights reserved.
        </div>
      </div>
    </footer>
  );
}
