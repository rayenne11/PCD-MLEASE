import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  Eye,
  EyeOff,
  ArrowRight,
  User,
  Lock,
  Mail,
  AlertCircle,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import "./AuthPage.css";

export default function AuthPage({ isSignIn: propIsSignIn }) {
  const location = useLocation();
  const [isSignIn, setIsSignIn] = useState(
    propIsSignIn ?? location.pathname === "/signin"
  );
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const navigate = useNavigate();
  const { signin, signup, isLoading, isAuthenticated } = useAuth();

  // Redirect if already authenticated, but only after loading is complete
  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      setTimeout(() => {
        navigate("/dashboard");
      }, 1000); // Add a small delay to ensure state is fully updated
    }
  }, [isLoading, isAuthenticated, navigate]);

  // Clear error when form changes
  useEffect(() => {
    setError("");
    setSuccess("");
  }, [isSignIn, email, password, name, username]);

  const toggleForm = () => {
    setIsSignIn(!isSignIn);
    setEmail("");
    setPassword("");
    setName("");
    setUsername("");
    setError("");
    setSuccess("");

    // Update URL without reload
    navigate(isSignIn ? "/signup" : "/signin", { replace: true });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    try {
      if (isSignIn) {
        // Sign In logic
        if (!username || !password) {
          throw new Error("Please enter your username and password");
        }

        const result = await signin(username, password);

        if (result.success) {
          setSuccess("Sign in successful! Redirecting...");
        } else {
          setError(result.error);
        }
      } else {
        // Sign Up logic
        if (!username || !email || !password || !name) {
          throw new Error("Please fill in all fields");
        }

        if (password.length < 6) {
          throw new Error("Password must be at least 6 characters");
        }

        const result = await signup(username, email, password, name);

        if (result.success) {
          setSuccess("Account created successfully! Redirecting...");
        } else {
          setError(result.error);
        }
      }
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-form-container">
        <div className="auth-form-header">
          <h2 className="auth-form-title">
            {isSignIn ? "Sign in to your account" : "Create a new account"}
          </h2>
          <div className="auth-form-subtitle">
            <p>
              {isSignIn
                ? "Don't have an account?"
                : "Already have an account?"}
              <button
                onClick={toggleForm}
                className="auth-toggle-button"
              >
                {isSignIn ? "Sign up" : "Sign in"}
              </button>
            </p>
          </div>
        </div>

        {error && (
          <div className="auth-error-message">
            <AlertCircle className="auth-error-icon" />
            <span className="auth-error-text">{error}</span>
          </div>
        )}

        {success && (
          <div className="auth-success-message">
            <AlertCircle className="auth-success-icon" />
            <span className="auth-success-text">{success}</span>
          </div>
        )}

        <div className="auth-form-body">
          {!isSignIn && (
            <div className="auth-form-field">
              <label htmlFor="name" className="auth-form-label">
                Full Name
              </label>
              <div className="auth-form-input-wrapper">
                <div className="auth-form-icon">
                  <User className="auth-form-icon-svg" />
                </div>
                <input
                  id="name"
                  name="name"
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="auth-form-input"
                  placeholder="John Doe"
                  disabled={isLoading}
                />
              </div>
            </div>
          )}

          <div className="auth-form-field">
            <label htmlFor="username" className="auth-form-label">
              Username
            </label>
            <div className="auth-form-input-wrapper">
              <div className="auth-form-icon">
                <User className="auth-form-icon-svg" />
              </div>
              <input
                id="username"
                name="username"
                type="text"
                autoComplete="username"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="auth-form-input"
                placeholder="johndoe"
                disabled={isLoading}
              />
            </div>
          </div>

          {!isSignIn && (
            <div className="auth-form-field">
              <label htmlFor="email" className="auth-form-label">
                Email address
              </label>
              <div className="auth-form-input-wrapper">
                <div className="auth-form-icon">
                  <Mail className="auth-form-icon-svg" />
                </div>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="auth-form-input"
                  placeholder="you@example.com"
                  disabled={isLoading}
                />
              </div>
            </div>
          )}

          <div className="auth-form-field">
            <label htmlFor="password" className="auth-form-label">
              Password
            </label>
            <div className="auth-form-input-wrapper">
              <div className="auth-form-icon">
                <Lock className="auth-form-icon-svg" />
              </div>
              <input
                id="password"
                name="password"
                type={showPassword ? "text" : "password"}
                autoComplete={isSignIn ? "current-password" : "new-password"}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="auth-form-input auth-form-input-password"
                placeholder="••••••••"
                disabled={isLoading}
              />
              <div className="auth-form-password-toggle">
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="auth-form-toggle-password"
                >
                  {showPassword ? (
                    <EyeOff className="auth-form-icon-svg" />
                  ) : (
                    <Eye className="auth-form-icon-svg" />
                  )}
                </button>
              </div>
            </div>
          </div>

          {isSignIn && (
            <div className="auth-form-options">
              <div className="auth-form-remember">
                <input
                  id="remember-me"
                  name="remember-me"
                  type="checkbox"
                  className="auth-form-checkbox"
                />
                <label htmlFor="remember-me" className="auth-form-label-small">
                  Remember me
                </label>
              </div>

              <div className="auth-form-forgot">
                <a href="#" className="auth-form-forgot-link">
                  Forgot your password?
                </a>
              </div>
            </div>
          )}

          <div>
            <button
              onClick={handleSubmit}
              className="auth-form-submit-button"
              disabled={isLoading}
            >
              {isLoading ? (
                <svg
                  className="auth-form-loading-spinner"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="auth-form-spinner-circle"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="auth-form-spinner-path"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              ) : (
                <>
                  <span className="auth-form-submit-icon">
                    <ArrowRight className="auth-form-icon-svg auth-form-submit-icon-svg" />
                  </span>
                  {isSignIn ? "Sign in" : "Sign up"}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}