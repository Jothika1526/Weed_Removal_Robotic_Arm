import streamlit as st


st.title('Precision Weed Removal using OpenManipulator-X')

# Sidebar Navigation
st.sidebar.title('Table of Contents')

# Define the sections for navigation (Table of Contents removed)
sections = [
    '1. Abstract',
    '2. Introduction',
    '3. Literature Review / Related Work',
    '4. Methodology & Implementation',
    '5. Results and Discussion',
    '6. Demo of Simulation and (or) hardware',
    '7. Conclusion and Future Work',
    '8. References'
]

# Create buttons for each section
buttons = {section: st.sidebar.button(section) for section in sections}

# Content for each section
if buttons['1. Abstract']:
    st.title('1. Abstract')
    st.write("""
    This website is built for our course project titled **"OpenManipulator-X Robotic Arm – A ROS2-based Manipulation System"**.
    In this project, we demonstrate the capabilities of the OpenManipulator-X robotic arm using ROS2 and Gazebo simulation.
    
    🚀 Check out our GitHub repo: [OpenManipulator_X_ROS2_Simulation](https://github.com/jothika2004/OpenManipulator_X_ROS2_Simulation.git)
    """)

elif buttons['2. Introduction']:
    st.title('2. Introduction')
    st.write('Content will be added soon...')

elif buttons['3. Literature Review / Related Work']:
    st.title('3. Literature Review / Related Work')
    st.write('Content will be added soon...')

elif buttons['4. Methodology & Implementation']:
    st.subheader('4. Methodology & Implementation')
    st.write("""
    ### Phase 1: Simulation Setup

    The initial phase involves setting up a virtual agricultural environment to test and refine the weed removal strategy using the OpenManipulator-X in a simulated scenario.

    #### Platform
    - **Gazebo with ROS2 Integration**: The simulation is conducted in Gazebo, a widely-used robotics simulator, integrated with ROS2 for real-time control, sensor data processing, and communication between components.

    #### Robot Setup
    - **URDF Integration**: The OpenManipulator-X URDF is loaded into the Gazebo environment, ensuring accurate replication of its kinematics and structure. The manipulator arm responds correctly to motion commands.
    """)

elif buttons['5. Results and Discussion']:
    st.title('5. Results and Discussion')
    st.write('Content will be added soon...')

elif buttons['6. Demo of Simulation and (or) hardware']:
    st.title('6. Demo of Simulation and (or) hardware')
    st.write('Content will be added soon...')

elif buttons['7. Conclusion and Future Work']:
    st.title('7. Conclusion and Future Work')
    st.write('Content will be added soon...')

elif buttons['8. References']:
    st.title('8. References')
    st.write('Content will be added soon...')
