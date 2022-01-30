import React from 'react';
import styles from "./Modal.module.css";

function Modal(props) {
  return (
    <div className={styles.modal}>
      <h1 className={styles.h1}>{props.name}</h1>
    </div>);
}

export default Modal;
