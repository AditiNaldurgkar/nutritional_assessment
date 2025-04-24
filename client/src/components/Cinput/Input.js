import React from 'react'
import "./Input.css"

function Input({label,placeholder,type,onChange}) {
  return (
    <div>
        <label id={label} className='label'>{label}</label>
        <input type={type} placeholder={placeholder} className='inp-box' onChange={onChange}></input>
    </div>
  )
}

export default Input