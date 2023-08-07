import "./Navbar.css";
import { useStore } from "../../helpers/store";
import { useEffect } from "react";
import { storeData } from "../../helpers/local-storage";

function Navbar() {
  const handleReset = useStore((state) => state.handleReset);
  const defaultModel = useStore((state) => state.defaultModel);
  const setDefaultModel = useStore((state) => state.setDefaultModel);

  const handleModelType = (e) => {
    e.preventDefault();
    setDefaultModel(!defaultModel);
  };

  useEffect(() => {
    storeData("defaultModel", defaultModel);
  }, [defaultModel]);

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
          <li className="nav-item">
            {defaultModel && (
              <a className="nav-link" href="/" onClick={handleModelType}>
                Default model
              </a>
            )}
            {!defaultModel && (
              <a className="nav-link" href="/" onClick={handleModelType}>
                Customize model
              </a>
            )}
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
