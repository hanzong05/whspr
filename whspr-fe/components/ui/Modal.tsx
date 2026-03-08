import { ReactNode } from "react";

const widthMap = { sm: "max-w-sm", md: "max-w-md", lg: "max-w-lg" };

type ModalProps = {
  onClose: () => void;
  children: ReactNode;
  maxWidth?: keyof typeof widthMap;
};

export default function Modal({ onClose, children, maxWidth = "md" }: ModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className={`relative bg-white rounded-2xl shadow-2xl w-full ${widthMap[maxWidth]} mx-4 overflow-hidden flex flex-col`}>
        {children}
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

Modal.Header = function ModalHeader({
  title,
  description,
  onClose,
}: {
  title: string;
  description?: string;
  onClose: () => void;
}) {
  return (
    <div className="flex items-start justify-between px-6 pt-6 pb-4 border-b border-gray-100">
      <div>
        <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
        {description && <p className="text-xs text-gray-400 mt-0.5">{description}</p>}
      </div>
      <button
        onClick={onClose}
        className="p-1 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors ml-4 flex-shrink-0"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
};

Modal.Body = function ModalBody({ children }: { children: ReactNode }) {
  return <div className="px-6 py-5 space-y-4">{children}</div>;
};

Modal.Footer = function ModalFooter({ children }: { children: ReactNode }) {
  return (
    <div className="px-6 py-4 border-t border-gray-100 flex gap-3">
      {children}
    </div>
  );
};
