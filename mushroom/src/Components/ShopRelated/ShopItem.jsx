import React, {useState} from 'react'

function Item(props) {
    const [qty, setQty] = useState(0)
    console.log(qty)
    const onQtyChange = (e, val) => {
        setQty(parseInt(e.target.value)) /*Sets quantity based on the props variable*/
    }
    return (
        <li className="indv">
            <img src={props.item.image} />
            <h2>{props.item.name}</h2>
            <h3>{props.item.price}</h3>
            <h4>{props.item.desc}</h4>
            <h3>
                Select Quantity:
                <input type="number" min="1" max="10" value={qty} onChange={onQtyChange} />
            </h3>
            <h3>
                <input type="button" value="Add Quantity to Cart" onClick={() => props.addCart(qty)} />
            </h3>
        </li>
    )
}

export default Item