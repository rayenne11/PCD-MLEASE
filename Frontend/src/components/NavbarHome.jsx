import mlease_logo from "../assets/logo.svg";
import "./NavbarHome.css";

const NavbarHome = () => {
  return (
    <nav className="navbar-home">
      <div className="navbar-home-left">
        <img
          className="navbar-home-logo"
          src={mlease_logo}
          alt=""
          width={130}
        />
        <div className="navbar-home-links">
          <a
            href="#"
            className="navbar-home-link"
          >
            Products
          </a>
          <a href="#" className="navbar-home-link">
            Solutions
          </a>
          <a href="#" className="navbar-home-link">
            Resources
          </a>
        </div>
      </div>
      <div className="navbar-home-right">
        <a
          href="/signin"
          className="navbar-home-login"
        >
          Login
        </a>
        <a
          href="/signin"
          className="navbar-home-get-started"
        >
          Get Started Free →
        </a>
      </div>
    </nav>
  );
};

export default NavbarHome;