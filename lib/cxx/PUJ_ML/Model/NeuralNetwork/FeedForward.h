// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__FeedForward__h__
#define __PUJ_ML__Model__NeuralNetwork__FeedForward__h__

#include <functional>
#include <string>
#include <vector>
#include <PUJ_ML/Model/Base.h>
#include <PUJ_ML/Model/NeuralNetwork/Activations.h>

namespace PUJ_ML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _R >
      class FeedForward
        : public PUJ_ML::Model::Base< PUJ_ML::Model::NeuralNetwork::FeedForward< _R >, _R >
      {
      public:
        using Self = FeedForward;
        using Superclass =
          PUJ_ML::Model::Base< PUJ_ML::Model::NeuralNetwork::FeedForward< _R >, _R >;

        using TReal = typename Superclass::TReal;
        using TMatrix = typename Superclass::TMatrix;
        using TCol = typename Superclass::TCol;
        using TRow = typename Superclass::TRow;

        using MMatrix = Eigen::Map< TMatrix >;
        using MCol = Eigen::Map< TCol >;
        using MRow = Eigen::Map< TRow >;

        using TActivationSignature = void( TMatrix&, const TMatrix&, bool );
        using TActivation = std::function< TActivationSignature >;
        using TBaseActivations =
          PUJ_ML::Model::NeuralNetwork::Activations< Self >;

      public:
        FeedForward( const unsigned long long& n = 1 );
        virtual ~FeedForward( );

        unsigned long long number_of_inputs( ) const;
        unsigned long long number_of_inputs( const unsigned long long& l ) const;
        void init( const unsigned long long& n = 0 );

        void add_layer(
          const unsigned long long& input,
          const unsigned long long& output,
          const std::string& activation
          );
        void add_layer(
          const unsigned long long& output,
          const std::string& activation
          );

        template< class _Y, class _X >
        void evaluate(
          Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
          ) const;

        template< class _Y, class _X >
        void threshold(
          Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
          ) const;

      protected:

        template< class _Y, class _X >
        void _evaluate(
          Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X,
          std::vector< TMatrix >* As = nullptr,
          std::vector< TMatrix >* Zs = nullptr
          ) const;

        void _reassign_memory( );

        virtual void _from_stream( std::istream& i ) override;
        virtual void _to_stream( std::ostream& o ) const override;

      protected:
        TBaseActivations s_Activations;

        std::vector< MMatrix >     m_W;
        std::vector< MRow >        m_B;
        std::vector< TActivation > m_A;
        std::vector< std::string > m_S;

      public:
        /**
         */
        class Cost
        {
        public:
          using Self = Cost;
          using TModel = typename Superclass::Derived;

        public:
          Cost( TModel* m );
          virtual ~Cost( ) = default;

          template< class _X, class _Y >
          TReal evaluate(
            const Eigen::EigenBase< _X >& X,
            const Eigen::EigenBase< _Y >& Y
            ) const;

          template< class _X, class _Y >
          TReal gradient(
            std::vector< TReal >& G,
            const Eigen::EigenBase< _X >& X,
            const Eigen::EigenBase< _Y >& Y
            ) const;

        protected:
          TModel* m_Model;
        };
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __PUJ_ML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
