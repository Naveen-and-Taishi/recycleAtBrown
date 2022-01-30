import styles from './App.module.css';
import Navbar from './Components/Navbar/Navbar';
import Dectector from "./Components/Detector/Detector";
import Modal from './Components/Modal/Modal';
import Popup from 'reactjs-popup';
import 'reactjs-popup/dist/index.css';
import { useEffect, useState } from 'react';

function App() {

  const [isOpen, setIsOpen] = useState(false)
  const [category, setCategory] = useState(0)

  function makePrediction() {
    const res = 0;
    // whatever tensorflow stuff
    setIsOpen(!isOpen)
    setCategory(res)
  }

  function runModalHandler(e) {
    console.log("GO");
  };

  return (
    <div className={styles.App}>
      <Navbar />
      <Dectector/>
      {isOpen && <Modal name={category}/>}
    </div>
  );
}

export default App;
