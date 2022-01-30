import styles from './App.module.css';
import Navbar from './Components/Navbar/Navbar';
import Dectector from "./Components/Detector/Detector";

function App() {
  return (
    <div className={styles.App}>
      <Navbar />
      <Dectector/>
    </div>
  );
}

export default App;
