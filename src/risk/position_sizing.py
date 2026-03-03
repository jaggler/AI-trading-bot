def calculate_lot_size(account_balance, risk_percentage=0.01, stop_loss_pips=10, pip_value=10):
    """
    Calculate the position lot size based on a fixed risk percentage.

    Args:
        account_balance (float): Total account balance.
        risk_percentage (float): Percentage of account to risk (e.g., 0.01 for 1%).
        stop_loss_pips (float): Stop loss in pips.
        pip_value (float): Value of a pip per standard lot.

    Returns:
        float: Calculated lot size.
    """
    risk_amount = account_balance * risk_percentage
    # Formula: Risk Amount / (Stop Loss in Pips * Pip Value per Lot)
    lot_size = risk_amount / (stop_loss_pips * pip_value)

    # Return lot size rounded to 2 decimal places (standard for many brokers)
    return round(lot_size, 2)
