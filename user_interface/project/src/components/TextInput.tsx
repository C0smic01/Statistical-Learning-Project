import React from 'react';

interface TextInputProps {
  maxLength: number;
  onTextChange: (text: string) => void;
  value: string;
}

const TextInput: React.FC<TextInputProps> = ({ maxLength, onTextChange, value }) => {
  const remainingChars = maxLength - value.length;

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newText = e.target.value;
    if (newText.length <= maxLength) {
      onTextChange(newText);
    }
  };

  return (
    <div className="w-full">
      <label htmlFor="text-input" className="block text-sm font-medium text-gray-300 mb-1">
        Enter text to analyze
      </label>
      <textarea
        id="text-input"
        className="w-full p-3 bg-[#162B45] border border-[#1E293B] rounded-lg 
                 focus:outline-none focus:border-[#00D8FF] transition-colors 
                 resize-none text-white placeholder-gray-400"
        placeholder="Type something to analyze the emotion..."
        rows={4}
        value={value}
        onChange={handleChange}
      />
      <div className="flex justify-end mt-1">
        <span 
          className={`text-sm ${
            remainingChars <= 20 
              ? remainingChars <= 10 
                ? 'text-red-400' 
                : 'text-amber-400' 
              : 'text-gray-400'
          }`}
        >
          {remainingChars} characters remaining
        </span>
      </div>
    </div>
  );
}

export default TextInput;