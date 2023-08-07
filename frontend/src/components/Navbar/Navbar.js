import "./Navbar.css";
import { useStore } from "../../helpers/store";

function Navbar() {
  const handleReset = useStore((state) => state.handleReset);

  return (
    <nav className="fixed-top navbar navbar-expand-lg bg-body-tertiary">
      <div className="collapse navbar-collapse" id="navbarNav">
        <span className="navbar-brand mb-0 mx-3 h1">FAQ Bot</span>
        <ul className="navbar-nav">
          <li className="nav-item">
            <a className="nav-link" href="/" onClick={handleReset}>
              Reset chat
            </a>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
